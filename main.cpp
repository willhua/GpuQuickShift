#include "Image.h"
#include "Exception.h"
#include <fstream>
#include "quickshift_common.h"


void write_image(image_t im, const char * filename)
{
  /********** Copy from matlab style **********/
  Image IMGOUT(im.K > 1 ? Image::RGB : Image::L, im.N2, im.N1);
  for(int k = 0; k < im.K; k++)
    for(int col = 0; col < im.N2; col++)
      for(int row = 0; row < im.N1; row++)
      {
        /* Row transpose */
        unsigned char * pt = IMGOUT.getPixelPt(col, im.N1-1-row);
        /* scale 0-255 */
        pt[k] = (unsigned char) (im.I[row + col*im.N1 + k*im.N1*im.N2]/32*255);
      }


  /********** Write image **********/
  std::ofstream ofs(filename, std::ios::binary);
  if (!ofs) {
      throw Exception("Could not open the file");
  }
  ofs<<IMGOUT;
}

image_t imseg(image_t im, int * flatmap)
{
  /********** Mean Color **********/
  float * meancolor = (float *) calloc(im.N1*im.N2*im.K, sizeof(float)) ;
  float * counts    = (float *) calloc(im.N1*im.N2, sizeof(float)) ;

  //统计每个模点包含的子点的数量，以及把所有子点的像素值都加到模点的位置上
  for (int p = 0; p < im.N1*im.N2; p++)
  {
    counts[flatmap[p]]++;
    for (int k = 0; k < im.K; k++)
      meancolor[flatmap[p] + k*im.N1*im.N2] += im.I[p + k*im.N1*im.N2];
  }

  //统计root点的数量，即模点的数量
  int roots = 0;
  for (int p = 0; p < im.N1*im.N2; p++)
  {
    if (flatmap[p] == p)
      roots++;
  }
  printf("Roots: %d\n", roots);

  //计算一个模点所关联的区域的的像素平均值，并且该平均值保存在模点的位置
  int nonzero = 0;
  for (int p = 0; p < im.N1*im.N2; p++)
  {
    if (counts[p] > 0)
    {
      nonzero++;
      for (int k = 0; k < im.K; k++)
        meancolor[p + k*im.N1*im.N2] /= counts[p];
    }
  }
  if (roots != nonzero)
    printf("Nonzero: %d\n", nonzero);
  assert(roots == nonzero);


  //把计算得到的模点的颜色值赋予给他相关联的像素点，该像素点的值是模点关联区域的像素值的平均值
  /********** Create output image **********/
  image_t imout = im;
  imout.I = (float *) calloc(im.N1*im.N2*im.K, sizeof(float));
  for (int p = 0; p < im.N1*im.N2; p++)
    for (int k = 0; k < im.K; k++)
      imout.I[p + k*im.N1*im.N2] = meancolor[flatmap[p] + k*im.N1*im.N2];

  free(meancolor);
  free(counts);

  return imout;
}

//相当于mean shiftfilter
int * map_to_flatmap(float * map, unsigned int size)
{
  /********** Flatmap **********/
  int *flatmap      = (int *) malloc(size*sizeof(int)) ;
  for (unsigned int p = 0; p < size; p++)
  {
    flatmap[p] = map[p];
  }

  bool changed = true;
  while (changed)
  {
    changed = false;
    for (unsigned int p = 0; p < size; p++)
    {
      //如果changed一直为false，那么说明flatmap[p] != flatmap[flatmap[p]]永远不成立，即flatmap[p] == flatmap[flatmap[p]]
      //恒成立，那么就说明：已经用高概率的把低概率的点都替换掉了，即相当于的meanshift的filter已经完成
      changed = changed || (flatmap[p] != flatmap[flatmap[p]]);
      flatmap[p] = flatmap[flatmap[p]];
    }
  }

  //保证已经填平，即用高概率的点覆盖了低概率的点
  /* Consistency check */
  for (unsigned int p = 0; p < size; p++)
    assert(flatmap[p] == flatmap[flatmap[p]]);

  return flatmap;
}

void image_to_matlab(Image & IMG, image_t & im)
{
  /********** Convert image to MATLAB style representation **********/
  im.N1 = IMG.getHeight();
  im.N2 = IMG.getWidth();
  im.K  = IMG.getPixelSize();
  im.I = (float *) calloc(im.N1*im.N2*im.K, sizeof(float));
  for(int k = 0; k < im.K; k++)
    for(int col = 0; col < im.N2; col++)
      for(int row = 0; row < im.N1; row++)
      {
        unsigned char * pt = IMG.getPixelPt(col, im.N1-1-row);
        im.I[row + col*im.N1 + k*im.N1*im.N2] = 32. * pt[k] / 255.; // Scale 0-32
      }
}

int process(char * path )
{
  //Use command-line specified CUDA device, otherwise use device with highest Gflops/s
  float sigma = 6, tau = 10;
  char * file = "flowers2.pnm";
  char * mode = "gpu";
  char * outover = NULL;
  char * tstr; float tmp;
    file = path;
    mode = tstr;
    outover = tstr;
  char * modes[1];
  modes[0] = mode;
  int nmodes = 1;


  /********** Read image **********/
  Image IMG;
  char outfile[1024];

  std::ifstream ifs(file, std::ios::binary);
  if (!ifs) {
      throw Exception("Could not open the file");
  }
  ifs>>IMG;
  image_t im;

  image_to_matlab(IMG, im);

  /********** CUDA setup **********/
  unsigned int timer;


  float *map, *E, *gaps;
  int * flatmap;
  image_t imout;

  map          = (float *) calloc(im.N1*im.N2, sizeof(float)) ;
  gaps         = (float *) calloc(im.N1*im.N2, sizeof(float)) ;
  E            = (float *) calloc(im.N1*im.N2, sizeof(float)) ;

  for(int m = 0; m < nmodes; m++)
  {

    /********** Quick shift **********/
      quickshift(im, sigma, tau, map, gaps, E);

      //quickshift_gpu(im, sigma, tau, map, gaps, E);



    /* Consistency check */
    for(int p = 0; p < im.N1*im.N2; p++)
      if(map[p] == p) assert(gaps[p] == INF);

    flatmap = map_to_flatmap(map, im.N1*im.N2);
    imout = imseg(im, flatmap);
    
    /*
    sprintf(outfile, "%s", file);
    char * c = strrchr(outfile, '.');
    if(c) *c = '\0';
    sprintf(outfile, "%s-%s.pnm", outfile, modes[m]); 

    if(outover)
      write_image(imout, outover);
    else
      write_image(imout, outfile);

      */

    free(flatmap);
    free(imout.I);
  }


  /********** Cleanup **********/
  free(im.I);

  free(map);
  free(E);
  free(gaps);
}
