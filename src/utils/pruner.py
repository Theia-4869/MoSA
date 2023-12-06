#!/usr/bin/env python3
"""
Pruner class
"""
import torch


class Pruner:
    def __init__(self, cfg):
        self.cfg = cfg
        if cfg.MODEL.TRANSFER_TYPE == "adapter" or cfg.MODEL.TRANSFER_TYPE == "mosa":
            self.num = cfg.MODEL.ADAPTER.EXPERT_NUM
        elif cfg.MODEL.TRANSFER_TYPE == "lora" or cfg.MODEL.TRANSFER_TYPE == "mosl":
            self.num = cfg.MODEL.LORA.EXPERT_NUM
        self.sparsity = 1 / self.num

    def score(self, param):
        raise NotImplementedError
    
    def prune(self, score):
        k = int((1.0 - self.sparsity) * score.numel())
        threshold, _ = torch.kthvalue(torch.flatten(score), k)
        
        zero = torch.LongTensor([0]).to(score.device)
        one = torch.LongTensor([1]).to(score.device)
        return torch.where(score <= threshold, zero, one)

    # def divide(self, score):
    #     masks = []
    #     zero = torch.LongTensor([0]).to(score.device)
    #     one = torch.LongTensor([1]).to(score.device)
        
    #     for i in range(self.num):
    #         mask = torch.ones_like(score, dtype=torch.long)
            
    #         lower_k = int((self.sparsity * i) * score.numel()) + 1
    #         lower_threshold, _ = torch.kthvalue(torch.flatten(score), lower_k)
    #         mask = torch.where(score < lower_threshold, zero, mask)
            
    #         upper_k = int((self.sparsity * (i + 1)) * score.numel())
    #         upper_threshold, _ = torch.kthvalue(torch.flatten(score), upper_k)
    #         mask = torch.where(score > upper_threshold, zero, mask)

    #         masks.append(mask)
        
    #     return masks
    
    def divide(self, score, mode="all"):
        masks = []
        zero = torch.LongTensor([0]).to(score.device)
        one = torch.LongTensor([1]).to(score.device)
        
        for i in range(self.num):
            mask = torch.ones_like(score, dtype=torch.long)
            if self.cfg.MODEL.ADAPTER.BIAS and len(score.shape) == 1:
                masks.append(mask)
                continue
            
            if mode == "all":
                lower_k = int((self.sparsity * i) * score.numel()) + 1
                lower_threshold, _ = torch.kthvalue(torch.flatten(score), lower_k)
                mask = torch.where(score < lower_threshold, zero, mask)
                
                upper_k = int((self.sparsity * (i + 1)) * score.numel())
                upper_threshold, _ = torch.kthvalue(torch.flatten(score), upper_k)
                mask = torch.where(score > upper_threshold, zero, mask)
            
            elif mode == "row":
                r, c = score.shape
                flag = torch.zeros_like(score, dtype=torch.long)
                flag += torch.arange(r, device=score.device).view(r, 1)
                
                for j in range(r):
                    lower_k = int((self.sparsity * i) * score[j].numel()) + 1
                    lower_threshold, _ = torch.kthvalue(torch.flatten(score[j]), lower_k)
                    mask = torch.where((score < lower_threshold) & (flag == j), zero, mask)

                    upper_k = int((self.sparsity * (i + 1)) * score[j].numel())
                    upper_threshold, _ = torch.kthvalue(torch.flatten(score[j]), upper_k)
                    mask = torch.where((score > upper_threshold) & (flag == j), zero, mask)
            
            elif mode == "column":
                r, c = score.shape
                flag = torch.zeros_like(score, dtype=torch.long)
                flag += torch.arange(c, device=score.device)
                
                for j in range(c):
                    lower_k = int((self.sparsity * i) * score[:, j].numel()) + 1
                    lower_threshold, _ = torch.kthvalue(torch.flatten(score[:, j]), lower_k)
                    mask = torch.where((score < lower_threshold) & (flag == j), zero, mask)

                    upper_k = int((self.sparsity * (i + 1)) * score[:, j].numel())
                    upper_threshold, _ = torch.kthvalue(torch.flatten(score[:, j]), upper_k)
                    mask = torch.where((score > upper_threshold) & (flag == j), zero, mask)

            masks.append(mask)
        
        return masks

class Rand(Pruner):
    def __init__(self, cfg):
        super(Rand, self).__init__(cfg)
    
    def score(self, param):
        return torch.rand_like(param)

