import torch
from torch.utils.data import DataLoader
from models.transformer import Transformer
from utils.loss import LabelSmoothingLoss
from utils.optimizer import TransformerOptimizer
from data.dataset import TranslationDataset, collate_fn
from data.vocab import Vocabulary
import argparse
import os
from tqdm import tqdm
import logging
from torch.utils.tensorboard import SummaryWriter

def train(args):
    """
    训练函数
    
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
    train_dataset = TranslationDataset(
        src_sentences=args.train_src,
        tgt_sentences=args.train_tgt,
        src_vocab=src_vocab,
        tgt_vocab=tgt_vocab,
        max_seq_length=args.max_seq_length
    )
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
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
        dropout=args.dropout
    ).to(device)
    
    # 创建损失函数
    criterion = LabelSmoothingLoss(
        smoothing=args.label_smoothing,
        ignore_index=tgt_vocab.pad_idx
    ).to(device)
    
    # 创建优化器
    optimizer = TransformerOptimizer(
        model=model,
        d_model=args.d_model,
        warmup_steps=args.warmup_steps,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        max_grad_norm=args.max_grad_norm
    )
    
    # 创建TensorBoard写入器
    writer = SummaryWriter(args.log_dir)
    
    # 训练循环
    global_step = 0
    for epoch in range(args.num_epochs):
        model.train()
        total_loss = 0
        
        # 使用tqdm创建进度条
        progress_bar = tqdm(train_loader, desc=f'Epoch {epoch + 1}/{args.num_epochs}')
        
        for batch_idx, (src, tgt) in enumerate(progress_bar):
            # 将数据移动到设备
            src = src.to(device)
            tgt = tgt.to(device)
            
            # 前向传播
            output = model(src, tgt[:, :-1])
            
            # 计算损失
            loss = criterion(output.contiguous().view(-1, output.size(-1)),
                           tgt[:, 1:].contiguous().view(-1))
            
            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.clip_gradients()
            optimizer.step()
            
            # 更新统计信息
            total_loss += loss.item()
            global_step += 1
            
            # 更新进度条
            progress_bar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'avg_loss': f'{total_loss / (batch_idx + 1):.4f}'
            })
            
            # 记录到TensorBoard
            writer.add_scalar('train/loss', loss.item(), global_step)
            writer.add_scalar('train/learning_rate', optimizer.optimizer.param_groups[0]['lr'], global_step)
            
        # 计算平均损失
        avg_loss = total_loss / len(train_loader)
        logger.info(f'Epoch {epoch + 1}/{args.num_epochs}, Average Loss: {avg_loss:.4f}')
        
        # 保存检查点
        if (epoch + 1) % args.save_interval == 0:
            checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.optimizer.state_dict(),
                'loss': avg_loss
            }
            torch.save(checkpoint, os.path.join(args.checkpoint_dir, f'checkpoint_epoch_{epoch + 1}.pt'))
            
    # 保存最终模型
    torch.save(model.state_dict(), os.path.join(args.checkpoint_dir, 'final_model.pt'))
    writer.close()
    
def main():
    """
    主函数
    """
    parser = argparse.ArgumentParser(description='Train Transformer model')
    
    # 数据参数
    parser.add_argument('--train_src', type=str, required=True, help='Path to source training data')
    parser.add_argument('--train_tgt', type=str, required=True, help='Path to target training data')
    parser.add_argument('--src_vocab_path', type=str, required=True, help='Path to source vocabulary')
    parser.add_argument('--tgt_vocab_path', type=str, required=True, help='Path to target vocabulary')
    parser.add_argument('--max_seq_length', type=int, default=512, help='Maximum sequence length')
    
    # 模型参数
    parser.add_argument('--d_model', type=int, default=512, help='Model dimension')
    parser.add_argument('--num_heads', type=int, default=8, help='Number of attention heads')
    parser.add_argument('--num_encoder_layers', type=int, default=6, help='Number of encoder layers')
    parser.add_argument('--num_decoder_layers', type=int, default=6, help='Number of decoder layers')
    parser.add_argument('--d_ff', type=int, default=2048, help='Feed forward dimension')
    parser.add_argument('--dropout', type=float, default=0.1, help='Dropout rate')
    
    # 训练参数
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--num_epochs', type=int, default=10, help='Number of epochs')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--warmup_steps', type=int, default=4000, help='Number of warmup steps')
    parser.add_argument('--label_smoothing', type=float, default=0.1, help='Label smoothing')
    parser.add_argument('--weight_decay', type=float, default=0.01, help='Weight decay')
    parser.add_argument('--max_grad_norm', type=float, default=1.0, help='Maximum gradient norm')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of data loading workers')
    
    # 保存参数
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints', help='Directory to save checkpoints')
    parser.add_argument('--log_dir', type=str, default='logs', help='Directory to save logs')
    parser.add_argument('--save_interval', type=int, default=1, help='Save interval in epochs')
    
    args = parser.parse_args()
    
    # 创建保存目录
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)
    
    # 开始训练
    train(args)
    
if __name__ == '__main__':
    main() 