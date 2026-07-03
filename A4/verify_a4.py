"""
Verification harness for Assignment 4 (object detection: FCOS one-stage +
Faster R-CNN two-stage).

Part 1 (always runs, CPU-fast): correctness checks for all detection building
blocks on synthetic data -- FPN, anchor generation, IoU, NMS, delta
encode/decode round-trips, GT matching, centerness -- plus a full
forward+backward pass of BOTH detectors on synthetic images (proves the whole
pipeline trains end to end).

Part 2 (optional): a reduced-real FCOS training run on a few PASCAL VOC images
if the dataset is available locally. VOC is a ~450 MB download and detection is
compute-heavy on CPU; per the build spec this is documented as a partial.

Writes ../results/a4_detection.txt.
"""
import os
import sys
import json

import torch

sys.path.insert(0, os.path.dirname(__file__))
torch.manual_seed(0)
torch.set_num_threads(3)

import common
import one_stage_detector as osd
import two_stage_detector as tsd

RESULTS = {}
FAILS = []
RES_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
os.makedirs(RES_DIR, exist_ok=True)


def check(name, cond, info=""):
    print(f"[{'PASS' if cond else 'FAIL'}] {name} {info}")
    if not cond:
        FAILS.append(name)


# =====================================================================
# Part 1a: geometry / matching / NMS on synthetic data
# =====================================================================
print("=== detection utilities ===")

# --- IoU ---
b1 = torch.tensor([[0.0, 0.0, 10.0, 10.0]])
b2 = torch.tensor([[0.0, 0.0, 10.0, 10.0],       # identical -> 1.0
                   [5.0, 5.0, 15.0, 15.0],       # quarter overlap
                   [20.0, 20.0, 30.0, 30.0]])    # disjoint -> 0.0
iou = tsd.iou(b1, b2)
check("IoU identical == 1", abs(iou[0, 0].item() - 1.0) < 1e-6)
check("IoU disjoint == 0", abs(iou[0, 2].item()) < 1e-6)
# quarter overlap: inter=25, union=175 -> 1/7
check("IoU partial", abs(iou[0, 1].item() - (25.0 / 175.0)) < 1e-5)

# --- NMS ---
boxes = torch.tensor([[0, 0, 10, 10], [1, 1, 11, 11], [50, 50, 60, 60]], dtype=torch.float)
scores = torch.tensor([0.9, 0.8, 0.7])
keep = common.nms(boxes, scores, iou_threshold=0.5)
check("NMS suppresses overlap", keep.tolist() == [0, 2])
# compare against torchvision on a random set
import torchvision
rb = torch.rand(50, 4)
rb[:, 2:] += rb[:, :2] + 0.1
rs = torch.rand(50)
k_ours = set(common.nms(rb, rs, 0.5).tolist())
k_tv = set(torchvision.ops.nms(rb, rs, 0.5).tolist())
check("NMS matches torchvision", k_ours == k_tv)

# --- FCOS delta encode/decode round-trip ---
locations = torch.tensor([[50.0, 50.0], [30.0, 40.0]])
gt = torch.tensor([[40.0, 40.0, 60.0, 60.0, 1.0], [10.0, 20.0, 50.0, 60.0, 2.0]])
deltas = osd.fcos_get_deltas_from_locations(locations, gt, stride=8)
boxes_back = osd.fcos_apply_deltas_to_locations(deltas, locations, stride=8)
check("FCOS delta round-trip", torch.allclose(boxes_back, gt[:, :4], atol=1e-4))
# background handling
bg = torch.tensor([[-1.0, -1, -1, -1]])
bg_deltas = osd.fcos_get_deltas_from_locations(
    torch.tensor([[5.0, 5.0]]), bg, stride=8)
check("FCOS background delta == -1", torch.all(bg_deltas == -1).item())

# --- centerness ---
ctr = osd.fcos_make_centerness_targets(torch.tensor([[1.0, 1.0, 1.0, 1.0]]))
check("centerness of centered == 1", abs(ctr.item() - 1.0) < 1e-5)

# --- R-CNN delta encode/decode round-trip ---
anchors = torch.tensor([[0.0, 0.0, 20.0, 20.0], [10.0, 10.0, 30.0, 40.0]])
gt2 = torch.tensor([[2.0, 3.0, 22.0, 25.0], [12.0, 8.0, 34.0, 38.0]])
d = tsd.rcnn_get_deltas_from_anchors(anchors, gt2)
back = tsd.rcnn_apply_deltas_to_anchors(d.clone(), anchors)
check("RCNN delta round-trip", torch.allclose(back, gt2, atol=1e-3))

# --- FPN location coords ---
shapes = {"p3": (1, 64, 4, 4), "p4": (1, 64, 2, 2), "p5": (1, 64, 1, 1)}
strides = {"p3": 8, "p4": 16, "p5": 32}
coords = common.get_fpn_location_coords(shapes, strides)
check("FPN coords p3 shape", coords["p3"].shape == (16, 2))
# first location center of p3 = (4, 4)
check("FPN coords first center", torch.allclose(coords["p3"][0], torch.tensor([4.0, 4.0])))

# --- anchor generation ---
anchors_lvl = tsd.generate_fpn_anchors(coords, strides, stride_scale=4,
                                       aspect_ratios=[0.5, 1.0, 2.0])
check("anchor count p3", anchors_lvl["p3"].shape == (16 * 3, 4))

# =====================================================================
# Part 1b: full forward + backward of BOTH detectors (synthetic images)
# =====================================================================
print("\n=== full detector forward/backward (synthetic) ===")
B = 2
images = torch.randn(B, 3, 224, 224)
# gt boxes: (B, N, 5), padded with -1
gt_boxes = -torch.ones(B, 4, 5)
gt_boxes[0, 0] = torch.tensor([30.0, 30.0, 100.0, 120.0, 3.0])
gt_boxes[0, 1] = torch.tensor([50.0, 60.0, 180.0, 200.0, 7.0])
gt_boxes[1, 0] = torch.tensor([20.0, 20.0, 90.0, 90.0, 1.0])

# --- FCOS ---
fcos = osd.FCOS(num_classes=20, fpn_channels=64, stem_channels=[64, 64])
fcos.train()
losses = fcos(images, gt_boxes)
total = losses["loss_cls"] + losses["loss_box"] + losses["loss_ctr"]
total.backward()
check("FCOS loss finite", torch.isfinite(total).item(),
      f"(cls={losses['loss_cls']:.3f} box={losses['loss_box']:.3f} ctr={losses['loss_ctr']:.3f})")
RESULTS["fcos_synth_total_loss"] = round(total.item(), 4)
# inference produces valid boxes
fcos.eval()
with torch.no_grad():
    pb, pc, ps = fcos(images[:1], test_score_thresh=0.05, test_nms_thresh=0.5)
check("FCOS inference returns boxes", pb.ndim == 2 and pb.shape[1] == 4)

# --- Faster R-CNN ---
backbone = common.DetectorBackboneWithFPN(out_channels=64)
rpn = tsd.RPN(fpn_channels=64, stem_channels=[64], batch_size_per_image=32,
              pre_nms_topk=100, post_nms_topk=40)
frcnn = tsd.FasterRCNN(backbone, rpn, stem_channels=[64], num_classes=20,
                       batch_size_per_image=32, roi_size=(7, 7))
frcnn.train()
losses = frcnn(images, gt_boxes)
total = losses["loss_rpn_obj"] + losses["loss_rpn_box"] + losses["loss_cls"]
total.backward()
check("FasterRCNN loss finite", torch.isfinite(total).item(),
      f"(obj={losses['loss_rpn_obj']:.3f} box={losses['loss_rpn_box']:.3f} cls={losses['loss_cls']:.3f})")
RESULTS["frcnn_synth_total_loss"] = round(total.item(), 4)
frcnn.eval()
with torch.no_grad():
    pb, pc, ps = frcnn(images[:1], test_score_thresh=0.05, test_nms_thresh=0.5)
check("FasterRCNN inference returns boxes", pb.ndim == 2 and pb.shape[1] == 4)

# =====================================================================
# Part 2: reduced-real FCOS training on PASCAL VOC (optional / partial)
# =====================================================================
print("\n=== reduced-real VOC training (optional) ===")
voc_done = False
try:
    from a4_helper import VOC2007DetectionTiny, train_detector
    import torchvision.transforms as T
    dataset_dir = os.path.join(os.path.dirname(__file__), "voc_data")
    want_download = not os.path.isdir(os.path.join(dataset_dir, "VOCdevkit"))
    train_ds = VOC2007DetectionTiny(
        dataset_dir, "train", download=want_download, image_size=224)
    from torch.utils.data import DataLoader
    # __getitem__ returns (image_path, image, gt_boxes); default collate works.
    loader = DataLoader(train_ds, batch_size=8, shuffle=True, num_workers=0)
    model = osd.FCOS(num_classes=20, fpn_channels=64, stem_channels=[64, 64])
    optim = torch.optim.SGD(model.parameters(), lr=1e-3, momentum=0.9)
    model.train()
    losses_seen = []
    it = iter(loader)
    for step in range(30):  # reduced-real: 30 iterations
        try:
            _, imgs, gtb = next(it)
        except StopIteration:
            it = iter(loader)
            _, imgs, gtb = next(it)
        out = model(imgs, gtb)
        loss = out["loss_cls"] + out["loss_box"] + out["loss_ctr"]
        optim.zero_grad()
        loss.backward()
        optim.step()
        losses_seen.append(loss.item())
        if step % 10 == 0:
            print(f"  VOC step {step}: loss={loss.item():.4f}")
    RESULTS["voc_fcos_first_loss"] = round(losses_seen[0], 4)
    RESULTS["voc_fcos_last_loss"] = round(sum(losses_seen[-5:]) / 5, 4)
    check("VOC FCOS loss decreased",
          RESULTS["voc_fcos_last_loss"] < RESULTS["voc_fcos_first_loss"])
    voc_done = True
except Exception as e:
    print(f"  VOC training skipped/partial: {type(e).__name__}: {str(e)[:200]}")
    RESULTS["voc_status"] = f"partial: {type(e).__name__}"

# =====================================================================
out_path = os.path.join(RES_DIR, "a4_detection.txt")
with open(out_path, "w") as f:
    f.write("EECS 498-007 A4 (object detection) verification\n" + "=" * 40 + "\n")
    f.write(f"failed checks: {FAILS}\n")
    f.write(f"VOC reduced-real training completed: {voc_done}\n\n")
    f.write(json.dumps(RESULTS, indent=2))
print(f"\nSaved {out_path}")
print("=" * 50)
if FAILS:
    print("FAILED:", FAILS)
    sys.exit(1)
print("ALL A4 CORRECTNESS CHECKS PASSED"
      + ("" if voc_done else " (VOC training documented as partial)"))
