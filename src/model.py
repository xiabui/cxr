"""Detector baseline dạng heatmap: backbone (torchvision) + decoder nhẹ.
Output 1 kênh heatmap [0,1] = xác suất tâm nốt. Thiết kế để dễ thay backbone
và cắm thêm attention/FPN ở giai đoạn sau.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision


def make_backbone(name="resnet34", pretrained=True, in_channels=1):
    weights = "DEFAULT" if pretrained else None
    net = getattr(torchvision.models, name)(weights=weights)
    # nhận in_channels kênh: khởi tạo conv1 mới, sao chép trọng số RGB trung bình
    w = net.conv1.weight.data  # [64,3,7,7]
    net.conv1 = nn.Conv2d(in_channels, 64, 7, 2, 3, bias=False)
    mean_w = w.mean(dim=1, keepdim=True)  # [64,1,7,7]
    # mỗi kênh đầu vào khởi tạo bằng trọng số trung bình -> giữ pretrained hữu ích
    net.conv1.weight.data = mean_w.repeat(1, in_channels, 1, 1)
    layers = nn.Sequential(net.conv1, net.bn1, net.relu, net.maxpool,
                           net.layer1, net.layer2, net.layer3, net.layer4)
    feat_ch = 512 if name in ("resnet18", "resnet34") else 2048
    return layers, feat_ch


class CBAM(nn.Module):
    """Convolutional Block Attention Module: chú ý theo kênh + theo không gian.
    Giúp mạng tập trung vào vùng tương phản thấp (nơi nốt khó ẩn) thay vì bị
    cấu trúc tương phản cao chi phối. Nhẹ, chèn sau đặc trưng backbone."""

    def __init__(self, channels, reduction=16):
        super().__init__()
        # chú ý theo kênh (channel attention)
        self.mlp = nn.Sequential(
            nn.Linear(channels, channels // reduction), nn.ReLU(True),
            nn.Linear(channels // reduction, channels))
        # chú ý theo không gian (spatial attention)
        self.spatial = nn.Conv2d(2, 1, 7, padding=3)

    def forward(self, x):
        b, c, _, _ = x.shape
        # channel attention: gộp avg + max pool toàn cục
        avg = F.adaptive_avg_pool2d(x, 1).view(b, c)
        mx = F.adaptive_max_pool2d(x, 1).view(b, c)
        ca = torch.sigmoid(self.mlp(avg) + self.mlp(mx)).view(b, c, 1, 1)
        x = x * ca
        # spatial attention: gộp avg + max theo kênh
        avg_s = x.mean(dim=1, keepdim=True)
        mx_s = x.max(dim=1, keepdim=True)[0]
        sa = torch.sigmoid(self.spatial(torch.cat([avg_s, mx_s], dim=1)))
        return x * sa


class UpBlock(nn.Module):
    def __init__(self, cin, cout):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(cin, cout, 3, padding=1), nn.BatchNorm2d(cout), nn.ReLU(True),
            nn.Conv2d(cout, cout, 3, padding=1), nn.BatchNorm2d(cout), nn.ReLU(True))

    def forward(self, x):
        return self.conv(F.interpolate(x, scale_factor=2, mode="bilinear",
                                       align_corners=False))


class HeatmapDetector(nn.Module):
    """backbone /32 -> up tới /stride. Với stride=4 cần 3 lần upsample."""

    def __init__(self, cfg):
        super().__init__()
        m = cfg["model"]
        self.in_channels = m.get("in_channels", 1)  # 1=baseline, 2=+bone suppression
        self.use_attention = m.get("use_attention", False)  # bật CBAM (đóng góp 2)
        self.backbone, ch = make_backbone(m["backbone"], m["pretrained"],
                                          in_channels=self.in_channels)
        self.attn = CBAM(ch) if self.use_attention else None
        ups, c = [], ch
        for out in (256, 128, 64):
            ups.append(UpBlock(c, out)); c = out
        self.decoder = nn.Sequential(*ups)  # /32 -> /4
        self.head = nn.Conv2d(c, 1, 1)

    def load_pretrained_backbone(self, ckpt_path, verbose=True):
        """Nạp trọng số backbone từ checkpoint pre-train NIH (giai đoạn 1).
        Chỉ nạp phần backbone; decoder/head detection giữ khởi tạo mới.
        Nếu input 2 kênh mà checkpoint 1 kênh: bỏ qua conv1 (giữ khởi tạo mới)."""
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        state = ckpt.get("backbone_state", ckpt)
        # xử lý lệch kênh conv1: checkpoint NIH là 1 kênh, detector có thể 2 kênh
        own = self.backbone.state_dict()
        filtered = {}
        for k, v in state.items():
            if k in own and own[k].shape == v.shape:
                filtered[k] = v
        missing, unexpected = self.backbone.load_state_dict(filtered, strict=False)
        if verbose:
            print(f"Đã nạp backbone từ {ckpt_path}"
                  + (f" (val AUC {ckpt['val_auc']:.4f})" if "val_auc" in ckpt else ""))
            skipped = [k for k in state if k not in filtered]
            if skipped:
                print(f"  bỏ qua {len(skipped)} key lệch shape "
                      f"(vd conv1 khi input {self.in_channels} kênh) — bình thường")
        return self

    def forward(self, x):
        f = self.backbone(x)
        if self.attn is not None:
            f = self.attn(f)        # CBAM chú ý trên đặc trưng backbone
        d = self.decoder(f)
        return torch.sigmoid(self.head(d))  # [B,1,H/4,W/4]


# ---------- Losses ----------
def focal_mse_loss(pred, target, alpha=2.0, beta=4.0, eps=1e-6, pos_weight=None,
                   weight_map=None):
    """Penalty-reduced focal loss (CornerNet/CenterNet-style) cho heatmap.
    pos_weight khuếch đại tín hiệu dương để model không 'trốn' bằng cách xuất
    toàn giá trị thấp. weight_map (đóng góp 3): nhân trọng số pos_loss theo
    subtlety — nốt khó (sub thấp) trọng số cao -> model 'cố' hơn ở nốt khó."""
    # clamp chặt hơn (1e-4) để log không ra giá trị cực lớn dưới fp16/AMP
    pred = pred.clamp(1e-4, 1 - 1e-4)
    pos_inds = target.eq(1).float()
    neg_inds = target.lt(1).float()
    neg_weights = torch.pow(1 - target, beta)

    pos_loss = torch.log(pred) * torch.pow(1 - pred, alpha) * pos_inds
    neg_loss = torch.log(1 - pred) * torch.pow(pred, alpha) * neg_weights * neg_inds

    # đóng góp 3: nhân trọng số theo subtlety vào phần dương
    if weight_map is not None:
        pos_loss = pos_loss * weight_map

    num_pos = pos_inds.sum()
    pos_loss = pos_loss.sum()
    neg_loss = neg_loss.sum()

    # khuếch đại pos: mặc định 40x để cân lại số lượng pixel âm áp đảo
    pw = 40.0 if pos_weight is None else pos_weight
    if num_pos == 0:
        return -neg_loss
    return -(pw * pos_loss + neg_loss) / num_pos


def build_loss(name):
    return {"focal_mse": focal_mse_loss}.get(name, focal_mse_loss)
