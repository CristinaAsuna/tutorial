"""学生关键机制接入真实 forward；不调用参考关键机制作为后备。"""
import torch
import torch.nn.functional as F
import reference_instructblip as ref
import lesson1_instruction_queries as l1
import lesson2_query_only_vision as l2
import lesson3_dual_tokenizer_prefix as l3
from lesson4_training_and_generation import training_step

class PracticeLayer(ref.InstructionQFormerLayer):
    def forward(self,tokens,num_queries,vision_features,valid_tokens):
        tokens=tokens+self.self_attn(self.norm1(tokens),key_padding_mask=valid_tokens)
        tokens=l2.query_only_cross_attention(tokens,num_queries,vision_features,self.cross_attn,self.norm2)
        return tokens+self.mlp(self.norm3(tokens))

class PracticeQFormer(ref.InstructionAwareQFormer):
    def __init__(self):
        super().__init__()
        self.layers=torch.nn.ModuleList([PracticeLayer(32,4,24) for _ in range(2)])
    def forward(self,vision_features,instruction_ids,instruction_attention_mask=None):
        if instruction_ids.ndim!=2 or vision_features.shape[0]!=instruction_ids.shape[0]:
            raise ValueError("matching instruction/image batches required")
        length=instruction_ids.shape[1]
        if not 0<length<=self.max_instruction_length: raise ValueError("invalid instruction length")
        if instruction_attention_mask is None: instruction_attention_mask=torch.ones_like(instruction_ids,dtype=torch.bool)
        if instruction_attention_mask.shape!=instruction_ids.shape or not instruction_attention_mask.bool().any(1).all():
            raise ValueError("each instruction needs valid token and matching mask")
        pos=torch.arange(length,device=instruction_ids.device).unsqueeze(0)
        text=self.word_embeddings(instruction_ids)+self.position_embeddings(pos)
        tokens,valid=l1.instruction_aware_queries(self.query_tokens,text,instruction_attention_mask)
        for layer in self.layers: tokens=layer(tokens,self.num_queries,vision_features,valid)
        return self.norm(tokens[:,:self.num_queries])

class PracticeInstructBlip(ref.InstructBlipForConditionalGeneration):
    def forward(self,pixel_values,instruction_ids,llm_prompt_ids,answer_ids,*,instruction_attention_mask=None,
                llm_prompt_attention_mask=None,answer_attention_mask=None):
        pm=torch.ones_like(llm_prompt_ids,dtype=torch.bool) if llm_prompt_attention_mask is None else llm_prompt_attention_mask.bool()
        am=torch.ones_like(answer_ids,dtype=torch.bool) if answer_attention_mask is None else answer_attention_mask.bool()
        self._validate_llm_inputs(llm_prompt_ids,answer_ids,pm,am)
        queries=self.encode_instruction_aware_queries(pixel_values,instruction_ids,instruction_attention_mask)
        embeds,mask,labels=l3.build_llm_prefix(queries,llm_proj=self.llm_proj,embed_tokens=self.llm.embed_tokens,
            prompt_ids=llm_prompt_ids,answer_ids=answer_ids,prompt_mask=pm,answer_mask=am)
        logits=self.llm(embeds,mask)
        loss=F.cross_entropy(logits[:,:-1].reshape(-1,logits.shape[-1]),labels[:,1:].reshape(-1),ignore_index=-100)
        return {"loss":loss,"logits":logits,"labels":labels,"visual_prefix":self.llm_proj(queries)}

    @torch.no_grad()
    def generate(self,pixel_values,instruction_ids,llm_prompt_ids,*,instruction_attention_mask=None,
                 llm_prompt_attention_mask=None,max_new_tokens=4):
        self.eval()
        pm=torch.ones_like(llm_prompt_ids,dtype=torch.bool) if llm_prompt_attention_mask is None else llm_prompt_attention_mask.bool()
        dummy=torch.ones(llm_prompt_ids.shape[0],1,dtype=torch.long,device=llm_prompt_ids.device)
        self._validate_llm_inputs(llm_prompt_ids,dummy,pm,torch.ones_like(dummy,dtype=torch.bool))
        queries=self.encode_instruction_aware_queries(pixel_values,instruction_ids,instruction_attention_mask)
        empty=llm_prompt_ids[:,:0]
        embeds,valid,_=l3.build_llm_prefix(queries,llm_proj=self.llm_proj,embed_tokens=self.llm.embed_tokens,
            prompt_ids=llm_prompt_ids,answer_ids=empty,prompt_mask=pm,answer_mask=torch.zeros_like(empty,dtype=torch.bool))
        outputs=[]
        for row in range(embeds.shape[0]):
            current=embeds[row:row+1,valid[row].bool()]
            generated=[]
            for _ in range(max_new_tokens):
                mask=torch.ones(current.shape[:2],dtype=torch.bool,device=current.device)
                next_id=self.llm(current,mask)[:,-1].argmax(-1)
                generated.append(next_id)
                current=torch.cat((current,self.llm.embed_tokens(next_id).unsqueeze(1)),1)
            outputs.append(torch.cat(generated))
        return torch.stack(outputs)

def build_toy_instructblip():
    return PracticeInstructBlip(ref.MockVisionEncoder(),PracticeQFormer(),ref.MockDecoderLM())
