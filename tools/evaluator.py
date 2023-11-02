from detectron2.evaluation import DatasetEvaluator
from detectron2.structures import Boxes, BoxMode, pairwise_iou

class YoloEvaluator(DatasetEvaluator):
  def reset(self):
    self._predictions = []

  def process(self, inputs, outputs):
    for input, output in zip(inputs, outputs):
        prediction = {"image_id": input["image_id"]}
        if "instances" in output:
            instances = output["instances"].to(self._cpu_device)
            prediction["instances"] = instances_to_yolo_json(instances, input["image_id"])
        if "proposals" in output:
            prediction["proposals"] = output["proposals"].to(self._cpu_device)
        if len(prediction) > 1:
            self._predictions.append(prediction)

  def evaluate(self):
    # save self.count somewhere, or print it, or return it.
    return {"count": self.count}

def instances_to_yolo_json(instances, img_id):
    
    num_instance = len(instances)
    if num_instance == 0:
        return []

    boxes = instances.pred_boxes.tensor.numpy()
    boxes = BoxMode.convert(boxes, BoxMode.XYXY_ABS, BoxMode.XYWH_ABS)
    boxes = boxes.tolist()
    scores = instances.scores.tolist()
    classes = instances.pred_classes.tolist()

    results = []
    for k in range(num_instance):
        result = {
            "image_id": img_id,
            "category_id": classes[k],
            "bbox": boxes[k],
            "score": scores[k],
        }
        results.append(result)
    return results
   