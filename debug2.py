import torch
import torch.nn as nn
import torch.utils.checkpoint
import functools

# 定义一个简单的Transformer层
class TransformerLayer(nn.Module):
    def __init__(self, d_model=512, nhead=8):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead)
        self.norm1 = nn.LayerNorm(d_model)
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.ReLU(),
            nn.Linear(d_model * 4, d_model)
        )
        self.norm2 = nn.LayerNorm(d_model)
        
    def forward(self, x):
        # 自注意力部分
        attn_output, _ = self.self_attn(x, x, x)
        x = x + attn_output
        x = self.norm1(x)
        
        # 前馈网络部分
        ff_output = self.feed_forward(x)
        x = x + ff_output
        x = self.norm2(x)
        return x

# 定义选择性检查点策略
def policy_fn(ctx, op, *args, **kwargs):
    # 只在前向传播时打印信息
    if not ctx.is_recompute:
        print(f"Op: {op.__class__.__name__}, Save tensor: {should_save(op)}")
    
    # 策略：只保存LayerNorm和Linear层的输出，丢弃其他中间结果
    return should_save(op)

def should_save(op):
    # 决定哪些操作的输出需要保存
    save_ops = (nn.LayerNorm, nn.Linear)
    return isinstance(op, save_ops)

# 创建模型和输入
model = TransformerLayer()
batch_size, seq_len, d_model = 32, 128, 512
x = torch.randn(seq_len, batch_size, d_model)

# 使用选择性检查点
def run_with_selective_checkpoint():
    # 创建上下文函数
    context_fn = functools.partial(
        torch.utils.checkpoint.create_selective_checkpoint_contexts, 
        policy_fn
    )
    
    # 定义要检查点化的函数
    def forward_fn(input_tensor):
        return model(input_tensor)
    
    # 应用检查点
    output = torch.utils.checkpoint.checkpoint(
        forward_fn,
        x,
        use_reentrant=False,
        context_fn=context_fn,
    )
    
    # 计算一些损失并反向传播
    loss = output.sum()
    loss.backward()
    
    return output, loss

# 运行并测量内存使用
print("Running with selective checkpointing...")
torch.cuda.reset_peak_memory_stats()  # 如果使用CUDA
output, loss = run_with_selective_checkpoint()
if torch.cuda.is_available():
    peak_memory = torch.cuda.max_memory_allocated() / (1024 ** 2)
    print(f"Peak memory usage: {peak_memory:.2f} MB")

print("Done!")