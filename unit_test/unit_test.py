import torch
import unittest
import argparse
import os
from Evaluator.visual_text_training import BERTModelModule  # 注意确保这个 import 正确指向你的模型模块


class TestLossFunction(unittest.TestCase):
    def setUp(self):
        """加载保存的测试数据"""
        parser = argparse.ArgumentParser()
        parser.add_argument('--seed', type=int, default=42, help="seed number used in the test_data filename")
        args, _ = parser.parse_known_args()

        test_file = f"../Evaluator/test_data_seed{args.seed}.pt"
        assert os.path.exists(test_file), f"{test_file} not found!"

        self.test_data = torch.load(test_file)
        self.device = "cpu"
        self.model = BERTModelModule(model_name="bert-base-uncased", use_classification_head=True).to(self.device)

    def test_loss_consistency(self):
        """确保相同输入得到相同的 loss 和输出"""
        batch = (
            self.test_data["images"],
            {k: v for k, v in self.test_data["hypo_premise_inputs"].items()},
            {k: v for k, v in self.test_data["hypo_update_inputs"].items()},
            self.test_data["update_types"]
        )

        # 使用 model 中的 compute_loss_and_scores 方法重新计算
        loss, logits_hypo_premise, logits_hypo_update, score_hypo_premise, score_hypo_update = self.model.compute_loss_and_scores(
            batch)

        # 验证 loss 是否一致
        self.assertTrue(torch.allclose(loss, self.test_data["loss"], atol=1e-6), "Loss mismatch!")

        # 验证 logits_hypo_premise 是否一致
        if logits_hypo_premise is not None and self.test_data["logits_hypo_premise"] is not None:
            self.assertTrue(torch.allclose(logits_hypo_premise, self.test_data["logits_hypo_premise"], atol=1e-6),
                            "Logits hypo_premise mismatch!")

        # 验证 logits_hypo_update 是否一致
        self.assertTrue(torch.allclose(logits_hypo_update, self.test_data["logits_hypo_update"], atol=1e-6),
                        "Logits hypo_update mismatch!")

        # 验证 score_hypo_premise 是否一致
        if score_hypo_premise is not None and self.test_data["score_hypo_premise"] is not None:
            self.assertTrue(torch.allclose(score_hypo_premise, self.test_data["score_hypo_premise"], atol=1e-6),
                            "Score hypo_premise mismatch!")

        # 验证 score_hypo_update 是否一致
        self.assertTrue(torch.allclose(score_hypo_update, self.test_data["score_hypo_update"], atol=1e-6),
                        "Score hypo_update mismatch!")

        print("All tests passed!")


if __name__ == "__main__":
    unittest.main()
