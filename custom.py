from PIL import Image
import test

im = Image.open('drawing-1.png').convert('L')
im_array = np.asarray(im).ravel()
im_array = torch.from_numpy(im_array).type(torch.FloatTensor)
test.showImage(im_array)

im_array = im_array.unsqueeze(0)
out = test.model(im_array)
_,pred = torch.max(out.data,1)
print(int(pred))

