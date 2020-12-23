from data.preprocess import DATASET_PATH

class AverageMeter(object):
  '''A handy class from the PyTorch ImageNet tutorial''' 
  def __init__(self):
    self.reset()
  def reset(self):
    self.val, self.avg, self.sum, self.count = 0, 0, 0, 0
  def update(self, val, n=1):
    self.val = val
    self.sum += val * n
    self.count += n
    self.avg = self.sum / self.count

def show_results():
  # Show images 
  import matplotlib.pyplot as plt
  import matplotlib.image as mpimg
  image_pairs = [('outputs/color/img-0-epoch-0.jpg', 'outputs/gray/img-0-epoch-0.jpg'),
                ('outputs/color/img-1-epoch-0.jpg', 'outputs/gray/img-1-epoch-0.jpg'),
                ('outputs/color/img-2-epoch-0.jpg', 'outputs/gray/img-2-epoch-0.jpg'),
                ('outputs/color/img-3-epoch-0.jpg', 'outputs/gray/img-3-epoch-0.jpg'),
                ('outputs/color/img-4-epoch-0.jpg', 'outputs/gray/img-4-epoch-0.jpg'),
                ('outputs/color/img-5-epoch-0.jpg', 'outputs/gray/img-5-epoch-0.jpg'),
                ('outputs/color/img-6-epoch-0.jpg', 'outputs/gray/img-6-epoch-0.jpg'),
                ('outputs/color/img-7-epoch-0.jpg', 'outputs/gray/img-7-epoch-0.jpg'),
                ('outputs/color/img-8-epoch-0.jpg', 'outputs/gray/img-8-epoch-0.jpg'),
                ('outputs/color/img-9-epoch-0.jpg', 'outputs/gray/img-9-epoch-0.jpg')]
  for c, g in image_pairs:
    color = mpimg.imread(DATASET_PATH + c)
    gray  = mpimg.imread(DATASET_PATH + g)
    f, axarr = plt.subplots(1, 2)
    f.set_size_inches(15, 15)
    axarr[0].imshow(gray, cmap='gray')
    axarr[1].imshow(color)
    axarr[0].axis('off'), axarr[1].axis('off')
    plt.show()