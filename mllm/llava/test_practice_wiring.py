import unittest
from unittest.mock import patch
import torch
import practice_llava as practice
import lesson1_vision_and_projector as l1
import lesson2_image_token_packing as l2
import lesson3_sft_loss as l3

class Wiring(unittest.TestCase):
    def test_encode(self):
        with patch.object(l1,'vision_to_llm_tokens',side_effect=RuntimeError('student')):
            with self.assertRaisesRegex(RuntimeError,'student'): practice.build_toy_llava().encode_images(torch.zeros(1,3,8,8))
    def test_pack(self):
        with patch.object(l2,'pack_one_image',side_effect=RuntimeError('student')):
            with self.assertRaisesRegex(RuntimeError,'student'): practice.build_toy_llava().pack_multimodal_inputs(torch.tensor([[1,-200,2]]),torch.zeros(1,4,32))
    def test_role_labels(self):
        with patch.object(l3,'assistant_only_labels',return_value=torch.tensor([[-100,-100,7,2]])) as fn:
            practice.build_sft_example([],[],[7])
            self.assertEqual(fn.call_args.kwargs['assistant_mask'].tolist(),[[False,False,True,True]])

    def test_batch_skeleton_calls_both_helpers(self):
        model=practice.build_toy_llava()
        ids=torch.tensor([[1,-200,2],[3,-200,4]])
        features=torch.randn(2,4,32)
        with patch.object(l2,'replace_image_sentinel',side_effect=[torch.ones(6,32),torch.full((6,32),2.)]) as replace:
            with patch.object(l2,'expand_image_supervision',side_effect=[(torch.ones(6,dtype=torch.bool),torch.full((6,),-100)),(torch.ones(6,dtype=torch.bool),torch.full((6,),4))]) as supervise:
                packed,valid,labels=l2.pack_one_image(ids,features,model.llm.embed_tokens,labels=ids)
                self.assertEqual(replace.call_count,2)
                self.assertEqual(supervise.call_count,2)
                self.assertEqual(tuple(packed.shape),(2,6,32))
                self.assertTrue(valid.all())
                self.assertEqual(labels[1].tolist(),[4]*6)

    def test_demo_mode_switch_does_not_leak_bindings(self):
        import run_llava_demo as demo
        original_example=demo.build_sft_example
        original_model=demo.build_toy_llava
        # Stop at constructor, sufficient to verify per-call mode selection.
        with patch.object(practice,'build_toy_llava',side_effect=RuntimeError('practice selected')):
            with self.assertRaisesRegex(RuntimeError,'practice selected'): demo.main('practice')
        with patch('reference_llava.build_toy_llava',side_effect=RuntimeError('reference selected')):
            with self.assertRaisesRegex(RuntimeError,'reference selected'): demo.main('reference')
        self.assertIs(demo.build_sft_example,original_example)
        self.assertIs(demo.build_toy_llava,original_model)

if __name__=='__main__': unittest.main()
