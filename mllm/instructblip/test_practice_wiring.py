import unittest
from unittest.mock import patch
import torch
import practice_instructblip as practice
import lesson1_instruction_queries as l1
import lesson2_query_only_vision as l2
import lesson3_dual_tokenizer_prefix as l3

class Wiring(unittest.TestCase):
    def test_queries(self):
        model=practice.build_toy_instructblip()
        with patch.object(l1,'instruction_aware_queries',side_effect=RuntimeError('student')):
            with self.assertRaisesRegex(RuntimeError,'student'): model.encode_instruction_aware_queries(torch.zeros(1,3,8,8),torch.tensor([[4,5]]))
    def test_cross(self):
        with patch.object(l2,'query_only_cross_attention',side_effect=RuntimeError('student')):
            with self.assertRaisesRegex(RuntimeError,'student'): practice.PracticeLayer(8,2,6)(torch.zeros(1,4,8),2,torch.zeros(1,3,6),torch.ones(1,4,dtype=torch.bool))
    def test_prefix(self):
        model=practice.build_toy_instructblip()
        with patch.object(model,'encode_instruction_aware_queries',return_value=torch.zeros(1,4,32)):
            with patch.object(l3,'build_llm_prefix',side_effect=RuntimeError('student')) as fn:
                with self.assertRaisesRegex(RuntimeError,'student'): model(torch.zeros(1,3,8,8),torch.tensor([[4,5]]),torch.tensor([[9,10]]),torch.tensor([[14,15]]))
                self.assertEqual(fn.call_args.kwargs['prompt_ids'].tolist(),[[9,10]])
                self.assertEqual(fn.call_args.kwargs['answer_ids'].tolist(),[[14,15]])

    def test_generation_prefix(self):
        model=practice.build_toy_instructblip()
        with patch.object(model,'encode_instruction_aware_queries',return_value=torch.zeros(1,4,32)):
            with patch.object(l3,'build_llm_prefix',side_effect=RuntimeError('student')) as fn:
                with self.assertRaisesRegex(RuntimeError,'student'):
                    model.generate(torch.zeros(1,3,8,8),torch.tensor([[4,5]]),torch.tensor([[9,10]]))
                self.assertEqual(tuple(fn.call_args.kwargs['answer_ids'].shape),(1,0))

    def test_prefix_skeleton_drops_padding_and_dispatches(self):
        queries=torch.zeros(2,2,4)
        embedding=torch.nn.Embedding(20,6)
        with patch.object(l3,'project_visual_queries',return_value=torch.zeros(2,2,6)) as project:
            with patch.object(l3,'concatenate_prefix_row',side_effect=[torch.ones(4,6),torch.ones(6,6)]) as join:
                with patch.object(l3,'prefix_answer_labels',side_effect=[torch.tensor([-100,-100,-100,14]),torch.tensor([-100,-100,-100,-100,15,16])]) as label:
                    embeds,mask,targets=l3.build_llm_prefix(queries,llm_proj=torch.nn.Linear(4,6),embed_tokens=embedding,
                        prompt_ids=torch.tensor([[9,0],[10,11]]),answer_ids=torch.tensor([[14,0],[15,16]]),
                        prompt_mask=torch.tensor([[1,0],[1,1]],dtype=torch.bool),answer_mask=torch.tensor([[1,0],[1,1]],dtype=torch.bool))
                    self.assertEqual(project.call_count,1)
                    self.assertEqual(join.call_args_list[0].args[1].shape[0],1)
                    self.assertEqual(label.call_args_list[0].args[2].tolist(),[14])
                    self.assertEqual(mask[0].tolist(),[True]*4+[False]*2)
                    self.assertEqual(targets[0].tolist(),[-100,-100,-100,14,-100,-100])

if __name__=='__main__': unittest.main()
