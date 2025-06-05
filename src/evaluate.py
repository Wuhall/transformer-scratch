import torch
from torch.utils.data import DataLoader
from models.transformer import Transformer
from data.dataset import TranslationDataset, collate_fn
from data.vocab import Vocabulary
import argparse
import os
import logging
from tqdm import tqdm
import numpy as np
from nltk.translate.bleu_score import corpus_bleu
import nltk
import json

def evaluate(args):
    """
    评估函数
    
    Args:
        args: 命令行参数
    """
    # 设置日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f'Using device: {device}')
    
    # 加载词汇表
    src_vocab = Vocabulary.load(args.src_vocab_path)
    tgt_vocab = Vocabulary.load(args.tgt_vocab_path)
    logger.info(f'Source vocabulary size: {len(src_vocab)}')
    logger.info(f'Target vocabulary size: {len(tgt_vocab)}')
    
    # 创建数据集
    eval_dataset = TranslationDataset(
        src_sentences=args.eval_src,
        tgt_sentences=args.eval_tgt,
        src_vocab=src_vocab,
        tgt_vocab=tgt_vocab,
        max_seq_length=args.max_seq_length
    )
    
    # 创建数据加载器
    eval_loader = DataLoader(
        eval_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=args.num_workers
    )
    
    # 创建模型
    model = Transformer(
        src_vocab_size=len(src_vocab),
        tgt_vocab_size=len(tgt_vocab),
        d_model=args.d_model,
        num_heads=args.num_heads,
        num_encoder_layers=args.num_encoder_layers,
        num_decoder_layers=args.num_decoder_layers,
        d_ff=args.d_ff,
        max_seq_length=args.max_seq_length,
        dropout=0.0  # 评估时不需要dropout
    ).to(device)
    
    # 加载模型权重
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.eval()
    
    # 下载nltk数据（如果还没有）
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt')
    
    # 评估
    references = []
    hypotheses = []
    total_loss = 0
    
    with torch.no_grad():
        for batch_idx, (src, tgt) in enumerate(tqdm(eval_loader, desc='Evaluating')):
            # 将数据移动到设备
            src = src.to(device)
            tgt = tgt.to(device)
            
            # 生成翻译
            output = model.generate(
                src,
                max_length=args.max_seq_length,
                beam_size=args.beam_size,
                length_penalty=args.length_penalty
            )
            
            # 将输出转换为文本
            for i in range(output.size(0)):
                # 获取参考翻译
                ref = tgt[i].cpu().numpy()
                ref = [tgt_vocab.idx2word[idx] for idx in ref if idx != tgt_vocab.pad_idx]
                ref = ' '.join(ref)
                references.append([ref.split()])  # BLEU需要列表的列表
                
                # 获取模型翻译
                hyp = output[i].cpu().numpy()
                hyp = [tgt_vocab.idx2word[idx] for idx in hyp if idx != tgt_vocab.pad_idx]
                hyp = ' '.join(hyp)
                hypotheses.append(hyp.split())
    
    # 计算BLEU分数
    bleu_score = corpus_bleu(references, hypotheses)
    logger.info(f'BLEU Score: {bleu_score:.4f}')
    
    # 保存评估结果
    results = {
        'bleu_score': bleu_score,
        'num_samples': len(references),
        'model_path': args.model_path,
        'beam_size': args.beam_size,
        'length_penalty': args.length_penalty
    }
    
    with open(os.path.join(args.output_dir, 'eval_results.json'), 'w') as f:
        json.dump(results, f, indent=4)
    
    # 保存一些示例翻译
    examples = []
    for i in range(min(10, len(references))):
        examples.append({
            'reference': ' '.join(references[i][0]),
            'hypothesis': ' '.join(hypotheses[i])
        })
    
    with open(os.path.join(args.output_dir, 'translation_examples.json'), 'w') as f:
        json.dump(examples, f, indent=4, ensure_ascii=False)
    
    logger.info(f'Evaluation results saved to {args.output_dir}')

def main():
    """
    主函数
    """
    parser = argparse.ArgumentParser(description='Evaluate Transformer model')
    
    # 数据参数
    parser.add_argument('--eval_src', type=str, required=True, help='Path to source evaluation data')
    parser.add_argument('--eval_tgt', type=str, required=True, help='Path to target evaluation data')
    parser.add_argument('--src_vocab_path', type=str, required=True, help='Path to source vocabulary')
    parser.add_argument('--tgt_vocab_path', type=str, required=True, help='Path to target vocabulary')
    parser.add_argument('--max_seq_length', type=int, default=512, help='Maximum sequence length')
    
    # 模型参数
    parser.add_argument('--d_model', type=int, default=512, help='Model dimension')
    parser.add_argument('--num_heads', type=int, default=8, help='Number of attention heads')
    parser.add_argument('--num_encoder_layers', type=int, default=6, help='Number of encoder layers')
    parser.add_argument('--num_decoder_layers', type=int, default=6, help='Number of decoder layers')
    parser.add_argument('--d_ff', type=int, default=2048, help='Feed forward dimension')
    
    # 评估参数
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--beam_size', type=int, default=5, help='Beam size for beam search')
    parser.add_argument('--length_penalty', type=float, default=0.6, help='Length penalty for beam search')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of data loading workers')
    
    # 其他参数
    parser.add_argument('--model_path', type=str, required=True, help='Path to model weights')
    parser.add_argument('--output_dir', type=str, default='eval_results', help='Directory to save evaluation results')
    
    args = parser.parse_args()
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 开始评估
    evaluate(args)
    
if __name__ == '__main__':
    main() 