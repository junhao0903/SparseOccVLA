import torch 
import torch.nn as nn
import torch.nn.functional as F
from .flash_attention import FlashAttention
import math
def normalize_bbox(bboxes):
    cx = bboxes[..., 0:1]
    cy = bboxes[..., 1:2]
    cz = bboxes[..., 2:3]
    w = bboxes[..., 3:4].log()
    l = bboxes[..., 4:5].log()
    h = bboxes[..., 5:6].log()
    rot = bboxes[..., 6:7]

    if bboxes.size(-1) > 7:
        vx = bboxes[..., 7:8]
        vy = bboxes[..., 8:9]
        out = torch.cat([cx, cy, w, l, cz, h, rot.sin(), rot.cos(), vx, vy], dim=-1)
    else:
        out = torch.cat([cx, cy, w, l, cz, h, rot.sin(), rot.cos()], dim=-1)

    return out


def denormalize_bbox(normalized_bboxes):
    rot_sin = normalized_bboxes[..., 6:7]
    rot_cos = normalized_bboxes[..., 7:8]
    rot = torch.atan2(rot_sin, rot_cos)

    cx = normalized_bboxes[..., 0:1]
    cy = normalized_bboxes[..., 1:2]
    cz = normalized_bboxes[..., 4:5]

    w = normalized_bboxes[..., 2:3].exp()
    l = normalized_bboxes[..., 3:4].exp()
    h = normalized_bboxes[..., 5:6].exp()

    if normalized_bboxes.size(-1) > 8:
        vx = normalized_bboxes[..., 8:9]
        vy = normalized_bboxes[..., 9:10]
        out = torch.cat([cx, cy, cz, w, l, h, rot, vx, vy], dim=-1)
    else:
        out = torch.cat([cx, cy, cz, w, l, h, rot], dim=-1)

    return out


def encode_bbox(bboxes, pc_range=None):
    xyz = bboxes[..., 0:3].clone()
    wlh = bboxes[..., 3:6].log()
    rot = bboxes[..., 6:7]

    if pc_range is not None:
        xyz[..., 0] = (xyz[..., 0] - pc_range[0]) / (pc_range[3] - pc_range[0])
        xyz[..., 1] = (xyz[..., 1] - pc_range[1]) / (pc_range[4] - pc_range[1])
        xyz[..., 2] = (xyz[..., 2] - pc_range[2]) / (pc_range[5] - pc_range[2])

    if bboxes.shape[-1] > 7:
        vel = bboxes[..., 7:9].clone()
        return torch.cat([xyz, wlh, rot.sin(), rot.cos(), vel], dim=-1)
    else:
        return torch.cat([xyz, wlh, rot.sin(), rot.cos()], dim=-1)


def decode_bbox(bboxes, pc_range=None):
    xyz = bboxes[..., 0:3].clone()
    wlh = bboxes[..., 3:6].exp()
    rot = torch.atan2(bboxes[..., 6:7], bboxes[..., 7:8])

    if pc_range is not None:
        xyz[..., 0] = xyz[..., 0] * (pc_range[3] - pc_range[0]) + pc_range[0]
        xyz[..., 1] = xyz[..., 1] * (pc_range[4] - pc_range[1]) + pc_range[1]
        xyz[..., 2] = xyz[..., 2] * (pc_range[5] - pc_range[2]) + pc_range[2]

    if bboxes.shape[-1] > 8:
        vel = bboxes[..., 8:10].clone()
        return torch.cat([xyz, wlh, rot, vel], dim=-1)
    else:
        return torch.cat([xyz, wlh, rot], dim=-1)


def encode_points(points, pc_range=None):
    points = points.clone()
    points[..., 0] = (points[..., 0] - pc_range[0]) / (pc_range[3] - pc_range[0])
    points[..., 1] = (points[..., 1] - pc_range[1]) / (pc_range[4] - pc_range[1])
    points[..., 2] = (points[..., 2] - pc_range[2]) / (pc_range[5] - pc_range[2])
    return points


def decode_points(points, pc_range=None):
    points = points.clone()
    points[..., 0] = points[..., 0] * (pc_range[3] - pc_range[0]) + pc_range[0]
    points[..., 1] = points[..., 1] * (pc_range[4] - pc_range[1]) + pc_range[1]
    points[..., 2] = points[..., 2] * (pc_range[5] - pc_range[2]) + pc_range[2]
    return points


def resort_points(raw_points,features):
    if raw_points.ndim==4:
        B,N,P,C = raw_points.shape
        points = raw_points.mean(-2)
    else:
        B,N,C = raw_points.shape
        points = raw_points
    assert C==3
    sorted_points = []
    sorted_features = []
    keys = points[...,0]*1e3 + points[...,1]*1e2+points[...,2]
    for b in range(B):
        idx = torch.argsort(keys[b],descending=True)
        sorted_points.append(raw_points[b][idx])
        if features is not None:
            sorted_features.append(features[b][idx])
    return torch.stack(sorted_points,dim=0),torch.stack(sorted_features,dim=0) if features is not None else None

def purify_points(points,ego2lidar):
    
    points = points[:,:3]
    ego2lidar = torch.from_numpy(ego2lidar).to(points)
    lidar2ego = torch.linalg.inv(ego2lidar)
    points_pad = torch.cat([points,torch.ones_like(points[:,:1])],dim=-1)
    points = torch.matmul(points_pad,lidar2ego.transpose(0,1))
    mask = (points[:,0]<40) & (points[:,0]>-40) & (points[:,1]>-40) & (points[:,1]<40) & (points[:,2]<5.4) & (points[:,2]>-1)
    return points[mask,:3].contiguous()


class CrossAttentionBlock(nn.Module):
    def __init__(self, dim: int,
                 num_heads: int = 16,
                 qkv_bias: bool = True,
                 qk_norm: bool = True,
                 attn_drop: float = 0.,
                 proj_drop: float = 0.,
                 norm_layer: nn.Module = nn.LayerNorm, ) -> None:
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        x_dtype = x.dtype
        B, N, C = x.shape
        q = self.q(x).reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)  # B, num_heads, N, head_dim
        kv = self.kv(y.to(x_dtype)).reshape(B, -1, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        k, v = kv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)
        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))
        attn = F.softmax(attn, dim=-1)
        x = attn @ v
        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x




class FFN(nn.Module):
    def __init__(self, dim: int,
                 mlp_ratio: int = 4,
                 drop: float = 0.) -> None:
        super().__init__()
        hidden_dim = int(dim * mlp_ratio)
        self.fc = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(drop)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)

class AttentionBlock(nn.Module):
    def __init__(self, dim: int,
                 num_heads: int = 16,
                 qkv_bias: bool = True,
                 qk_norm: bool = False,
                 attn_drop: float = 0.,
                 proj_drop: float = 0.,
                 norm_layer: nn.Module = nn.LayerNorm, ) -> None:
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        factory_kwargs = {'device': 'cuda', 'dtype': torch.float16}
        self.inner_attn = FlashAttention(attention_dropout=0.0, **factory_kwargs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_dtype = x.dtype
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 1, 3,
                                                                                  4)  # 3, B, N, num_heads, head_dim
        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)
        qkv = torch.stack([q, k, v], dim=2).to(torch.float16)
        x, _ = self.inner_attn(qkv, key_padding_mask=None, causal=False)
        x = x.reshape(B, N, C).to(x_dtype)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class TransformerCrossLayer(nn.Module):
    def __init__(self, dim: int,
                 num_heads: int = 16,
                 mlp_ratio: int = 4,
                 qkv_bias: bool = True,
                 qk_norm: bool = False,
                 drop: float = 0.,
                 attn_drop: float = 0.,
                 drop_path: float = 0.,
                 norm_layer: nn.Module = nn.LayerNorm) -> None:
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.norm2 = norm_layer(dim)
        self.norm3 = norm_layer(dim)
        self.cross_attn = CrossAttentionBlock(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_norm=True, attn_drop=attn_drop, proj_drop=drop)
        self.norm4 = norm_layer(dim)
        self.mlp = FFN(dim, mlp_ratio=mlp_ratio, drop=drop)
        self.self_attn = AttentionBlock(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_norm=qk_norm, attn_drop=attn_drop, proj_drop=drop)

    def forward(self, x: torch.Tensor, learnable_query: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        
        x = x + self.cross_attn(self.norm1(x), self.norm2(learnable_query))
        x = x + self.self_attn(self.norm3(x))
        x = x + self.mlp(self.norm4(x))
        # x = x.permute(0, 2, 1).reshape(B, C, H, W)
        return x

class EgoAttention(nn.Module):
    def __init__(self, dim: int,
                 num_layers: int,
                 num_heads: int = 16,
                 mlp_ratio: int = 4,
                 qkv_bias: bool = True,
                 qk_norm: bool = False,
                 drop: float = 0.,
                 attn_drop: float = 0.,
                 drop_path: float = 0.,
                 norm_layer: nn.Module = nn.LayerNorm,
                 ) -> None:
        super().__init__()
        self.num_layers = num_layers
        self.EgoAttention = nn.ModuleList()
        for i in range(num_layers):
            self.EgoAttention.append(
                TransformerCrossLayer(dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_norm=qk_norm,
                                      drop=drop, attn_drop=attn_drop, drop_path=drop_path, norm_layer=norm_layer))

    def forward(self, x: torch.Tensor, learnable_query: torch.Tensor) -> torch.Tensor:
        if self.num_layers == 0:
            return x
        else:
            for i, layer in enumerate(self.EgoAttention):
                x = layer(x, learnable_query)
            return x

class MLp(nn.Module):

    def __init__(self,
                 in_features,
                 hidden_features=None,
                 out_features=None,
                 act_layer=nn.ReLU,
                 drop=0.0):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop)
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop2 = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x

def gen_sineembed_for_position(pos_tensor, hidden_dim=256):
    """Mostly copy-paste from https://github.com/IDEA-opensource/DAB-DETR/
    """
    half_hidden_dim = hidden_dim // 2
    scale = 2 * math.pi
    dim_t = torch.arange(half_hidden_dim, dtype=torch.float32, device=pos_tensor.device)
    dim_t = 10000 ** (2 * (dim_t // 2) / half_hidden_dim)
    x_embed = pos_tensor[..., 0] * scale
    y_embed = pos_tensor[..., 1] * scale
    pos_x = x_embed[..., None] / dim_t
    pos_y = y_embed[..., None] / dim_t
    pos_x = torch.stack((pos_x[..., 0::2].sin(), pos_x[..., 1::2].cos()), dim=-1).flatten(-2)
    pos_y = torch.stack((pos_y[..., 0::2].sin(), pos_y[..., 1::2].cos()), dim=-1).flatten(-2)
    pos = torch.cat((pos_y, pos_x), dim=-1)
    return pos

def linear_relu_ln(embed_dims, in_loops, out_loops, input_dims=None):
    if input_dims is None:
        input_dims = embed_dims
    layers = []
    for _ in range(out_loops):
        for _ in range(in_loops):
            layers.append(nn.Linear(input_dims, embed_dims))
            layers.append(nn.ReLU(inplace=True))
            input_dims = embed_dims
        layers.append(nn.LayerNorm(embed_dims))
    return layers


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class ModulationLayer(nn.Module):

    def __init__(self, embed_dims: int, condition_dims: int):
        super(ModulationLayer, self).__init__()
        self.if_zeroinit_scale=False
        self.embed_dims = embed_dims
        self.scale_shift_mlp = nn.Sequential(
            nn.Mish(),
            nn.Linear(condition_dims, embed_dims*2),
        )
        self.init_weight()

    def init_weight(self):
        if self.if_zeroinit_scale:
            nn.init.constant_(self.scale_shift_mlp[-1].weight, 0)
            nn.init.constant_(self.scale_shift_mlp[-1].bias, 0)

    def forward(
        self,
        traj_feature,
        time_embed,
        global_cond=None,
        global_img=None,
    ):
        if global_cond is not None:
            global_feature = torch.cat([
                    global_cond, time_embed
                ], axis=-1)
        else:
            global_feature = time_embed
        if global_img is not None:
            global_img = global_img.flatten(2,3).permute(0,2,1).contiguous()
            global_feature = torch.cat([
                    global_img, global_feature
                ], axis=-1)
        
        scale_shift = self.scale_shift_mlp(global_feature)
        scale,shift = scale_shift.chunk(2,dim=-1)
        traj_feature = traj_feature * (1 + scale) + shift
        return traj_feature
class CustomTransformerDecoderLayer(nn.Module):
    def __init__(self, 
                 num_poses,
                 d_model,
                 d_ffn,
                 
                 ):
        super().__init__()
        self.dropout = nn.Dropout(0.1)
        self.dropout1 = nn.Dropout(0.1)
        self.num_poses = num_poses
        self.cross_query_attention = EgoAttention(d_model,num_layers=2,num_heads=8,mlp_ratio=2)
        self.cross_ego_attention = nn.MultiheadAttention(
            d_model,
            8,
            0.1,
            batch_first=True,
        )
        self.ffn = nn.Sequential(
            nn.Linear(d_ffn,d_ffn ),
            nn.ReLU(),
            nn.Linear(d_ffn, d_model),
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.time_modulation = ModulationLayer(d_model,1024)
        self.task_decoder = nn.Sequential(
            nn.Linear(d_model,d_model),
            nn.ReLU(),
            nn.Linear(d_model,d_model),
            nn.ReLU(),
            nn.Linear(d_model,6*2)
        )

    def forward(self, 
                traj_feature, 
                noisy_traj_points, 
                scene_query_feature,
                ego_query, 
                time_embed, 
                status_encoding,
                global_img=None):
        traj_feature = self.cross_query_attention(traj_feature,scene_query_feature)
        traj_feature = traj_feature + self.dropout(traj_feature)
        traj_feature = self.norm1(traj_feature)
        
        # traj_feature = traj_feature + self.dropout(self.self_attn(traj_feature, traj_feature, traj_feature)[0])
        # traj_feature = self.norm2(traj_feature)
        # 4.5 cross attention with  ego query
        traj_feature = traj_feature + self.dropout1(self.cross_ego_attention(traj_feature, ego_query,ego_query)[0])
        traj_feature = self.norm2(traj_feature)
        
        # 4.6 feedforward network
        traj_feature = self.norm3(self.ffn(traj_feature))
        # 4.8 modulate with time steps
        traj_feature = self.time_modulation(traj_feature, time_embed,global_cond=None,global_img=global_img)
        
        # 4.9 predict the offset & heading
        poses_reg = self.task_decoder(traj_feature).unflatten(-1,(-1,self.num_poses)) #bs,20,8,3; bs,20
        # poses_reg = poses_reg + noisy_traj_points
        

        return poses_reg