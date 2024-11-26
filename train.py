import argparse
from functools import partial

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau

from model import Im2LatexModel, Trainer
from utils import collate_fn, get_checkpoint
from data import Im2LatexDataset
from build_vocab import Vocab, load_vocab


def main():
    # 使用 argparse 定義程式參數，方便設定模型訓練時的超參數
    parser = argparse.ArgumentParser(description="Im2Latex Training Program")
    
    # 模型相關參數
    parser.add_argument("--emb_dim", type=int,
                        default=80, help="Embedding size")  # 詞嵌入的維度
    parser.add_argument("--dec_rnn_h", type=int, default=512,
                        help="The hidden state of the decoder RNN")  # 解碼器RNN的隱藏層維度
    parser.add_argument("--data_path", type=str,
                        default="./data/", help="The dataset's dir")  # 資料集所在的目錄
    parser.add_argument("--add_position_features", action='store_true',
                        default=False, help="Use position embeddings or not")  # 是否使用位置特徵
    
    # 訓練相關參數
    parser.add_argument("--max_len", type=int,
                        default=150, help="Max size of formula")  # 最大公式長度
    parser.add_argument("--dropout", type=float,
                        default=0., help="Dropout probability")  # Dropout 機率
    parser.add_argument("--cuda", action='store_true',
                        default=True, help="Use cuda or not")  # 是否使用 CUDA
    parser.add_argument("--batch_size", type=int, default=32)  # 每批次的數據量
    parser.add_argument("--epoches", type=int, default=15)  # 總訓練輪數
    parser.add_argument("--lr", type=float, default=3e-4,
                        help="Learning Rate")  # 初始學習率
    parser.add_argument("--min_lr", type=float, default=3e-5,
                        help="Minimum Learning Rate")  # 最小學習率
    parser.add_argument("--sample_method", type=str, default="teacher_forcing",
                        choices=('teacher_forcing', 'exp', 'inv_sigmoid'),
                        help="The method to schedule sampling")  # 選擇樣本方法
    parser.add_argument("--decay_k", type=float, default=1.,
                        help="Base of Exponential decay for Schedule Sampling. "
                        "When sample method is Exponential decay;"
                        "Or a constant in Inverse sigmoid decay Equation. "
                        "See details in https://arxiv.org/pdf/1506.03099.pdf"
                        )  # 採樣方法中的衰減參數

    parser.add_argument("--lr_decay", type=float, default=0.5,
                        help="Learning Rate Decay Rate")  # 學習率衰減比例
    parser.add_argument("--lr_patience", type=int, default=3,
                        help="Learning Rate Decay Patience")  # 學習率衰減耐受步數
    parser.add_argument("--clip", type=float, default=2.0,
                        help="The max gradient norm")  # 梯度裁剪的最大值
    parser.add_argument("--save_dir", type=str,
                        default="./ckpts", help="The dir to save checkpoints")  # 模型檢查點儲存路徑
    parser.add_argument("--print_freq", type=int, default=100,
                        help="The frequency to print message")  # 訓練過程中信息打印頻率
    parser.add_argument("--seed", type=int, default=2020,
                        help="The random seed for reproducing ")  # 隨機種子，確保結果可重現
    parser.add_argument("--from_check_point", action='store_true',
                        default=False, help="Training from checkpoint or not")  # 是否從檢查點繼續訓練

    args = parser.parse_args()
    max_epoch = args.epoches
    from_check_point = args.from_check_point

    # 如果指定從檢查點恢復，載入檢查點及參數
    if from_check_point:
        checkpoint_path = get_checkpoint(args.save_dir)
        checkpoint = torch.load(checkpoint_path)
        args = checkpoint['args']
    print("Training args:", args)

    # 設定隨機種子以保證結果可重現
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    # 建立詞彙表
    print("Load vocab...")
    vocab = load_vocab(args.data_path)

    # 確定設備 (CPU or GPU)
    use_cuda = True if args.cuda and torch.cuda.is_available() else False
    device = torch.device("cuda" if use_cuda else "cpu")

    # 建立資料加載器
    print("Construct data loader...")
    train_loader = DataLoader(
        Im2LatexDataset(args.data_path, 'train', args.max_len),
        batch_size=args.batch_size,
        collate_fn=partial(collate_fn, vocab.sign2id))  # 訓練資料集的加載器
    val_loader = DataLoader(
        Im2LatexDataset(args.data_path, 'validate', args.max_len),
        batch_size=args.batch_size,
        collate_fn=partial(collate_fn, vocab.sign2id))  # 驗證資料集的加載器

    # 初始化模型
    print("Construct model")
    vocab_size = len(vocab)
    model = Im2LatexModel(
        vocab_size, args.emb_dim, args.dec_rnn_h,
        add_pos_feat=args.add_position_features,
        dropout=args.dropout
    )
    model = model.to(device)  # 將模型移至指定設備
    print("Model Settings:")
    print(model)

    # 初始化優化器
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # 設定學習率調整策略
    lr_scheduler = ReduceLROnPlateau(
        optimizer,
        "min",
        factor=args.lr_decay,
        patience=args.lr_patience,
        verbose=True,
        min_lr=args.min_lr)

    # 如果從檢查點恢復，載入狀態
    if from_check_point:
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']
        lr_scheduler.load_state_dict(checkpoint['lr_sche'])
        # 初始化訓練器
        trainer = Trainer(optimizer, model, lr_scheduler,
                          train_loader, val_loader, args,
                          use_cuda=use_cuda,
                          init_epoch=epoch, last_epoch=max_epoch)
    else:
        trainer = Trainer(optimizer, model, lr_scheduler,
                          train_loader, val_loader, args,
                          use_cuda=use_cuda,
                          init_epoch=1, last_epoch=args.epoches)
    # 開始訓練
    trainer.train()


if __name__ == "__main__":
    main()