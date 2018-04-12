using UnityEngine;
using System.Collections;
using System.Collections.Generic;
using OpenCvSharp;

using Uk.Org.Adcock.Parallel;

public class LaserDotDetection : MonoBehaviour
{

    // Video parameters
    public MeshRenderer WebCamTextureRenderer;
    public MeshRenderer ProcessedTextureRenderer;
    public int deviceNumber;
    private WebCamTexture _webcamTexture;

    // Video size
    private const int imWidth = 800;
    private const int imHeight = 600;
    private int imFrameRate;

    // OpenCVSharp parameters
    private Mat videoSourceImage;
    private Mat cannyImage;
    private Texture2D processedTexture;
    private Vec3b[] videoSourceImageData;
    private byte[] cannyImageData;

    // Frame rate parameter
    private int updateFrameCount = 0;
    private int textureCount = 0;
    private int displayCount = 0;

    void Start() {

        WebCamDevice[] devices = WebCamTexture.devices;

            _webcamTexture = new WebCamTexture(devices[deviceNumber].name, imWidth, imHeight);

            WebCamTextureRenderer.material.mainTexture = _webcamTexture;

            _webcamTexture.Play();

            videoSourceImage = new Mat(imHeight, imWidth, MatType.CV_8UC3);
            videoSourceImageData = new Vec3b[imHeight * imWidth];
            cannyImage = new Mat(imHeight, imWidth, MatType.CV_8UC1);
            cannyImageData = new byte[imHeight * imWidth];

            processedTexture = new Texture2D(imWidth, imHeight, TextureFormat.RGBA32, true, true);

            ProcessedTextureRenderer.material.mainTexture = processedTexture;

        //Cv2.NamedWindow("Copy video");
        

    }


    
    void Update() {

        updateFrameCount++;

        if (_webcamTexture.isPlaying) {

            if (_webcamTexture.didUpdateThisFrame) {

                textureCount++;

                TextureToMat();
                UpdateWindow(videoSourceImage);
                ProcessImage(videoSourceImage);
                MatToTexture();
                DetectLaser();

            }

        }
        else {
            Debug.Log("Can't find camera!");
        }

        if (updateFrameCount % 30 == 0) {
            Debug.Log("Frame count: " + updateFrameCount + ", Texture count: " + textureCount + ", Display count: " + displayCount);
        }


    }


    void TextureToMat() {
        Color32[] c = _webcamTexture.GetPixels32();

        Parallel.For(0, imHeight, i => {
            for (var j = 0; j < imWidth; j++) {
                var col = c[j + i * imWidth];
                var vec3 = new Vec3b {
                    Item0 = col.b,
                    Item1 = col.g,
                    Item2 = col.r
                };
                videoSourceImageData[j + i * imWidth] = vec3;
            }
        });
        videoSourceImage.SetArray(0, 0, videoSourceImageData);
    }


    void MatToTexture() {
        cannyImage.GetArray(0, 0, cannyImageData);
        Color32[] c = new Color32[imHeight * imWidth];

        Parallel.For(0, imHeight, i => {
            for (var j = 0; j < imWidth; j++) {
                byte vec = cannyImageData[j + i * imWidth];
                var color32 = new Color32 {
                    r = vec,
                    g = vec,
                    b = vec,
                    a = 0
                };
                c[j + i * imWidth] = color32;
            }
        });

        processedTexture.SetPixels32(c);
        processedTexture.Apply();
    }


    void ProcessImage(Mat _image) {
        Cv2.Flip(_image, _image, FlipMode.X);
        Cv2.Canny(_image, cannyImage, 100, 100);
    }


    void UpdateWindow(Mat _image)
    {
        Cv2.Flip(_image, _image, FlipMode.X);
        displayCount++;
    }

    void DetectLaser()
    {
        Mat[] bgrChannels = new Mat[3];
        Cv2.Split(videoSourceImage, out bgrChannels);

        Mat rt = new Mat();
        Cv2.Threshold(bgrChannels[2], rt, 240.0, 255.0, ThresholdTypes.Binary);

        Cv2.GaussianBlur(rt, rt, new Size(5, 5), 1);
        rt *= 255;

        Mat labels = new Mat();
        Mat stats = new Mat();
        Mat centroids = new Mat();
        Cv2.ConnectedComponentsWithStats(rt, labels, stats, centroids);

        if (centroids.Rows > 1)
        {
            Cv2.Circle(videoSourceImage, (int)centroids.At<double>(1, 0), (int)centroids.At<double>(1, 1), 20, new Scalar(255, 255, 255));
            Cv2.Circle(cannyImage, (int)centroids.At<double>(1, 0), (int)centroids.At<double>(1, 1), 20, new Scalar(255, 255, 255));
        }

        Cv2.Flip(videoSourceImage, videoSourceImage, FlipMode.XY);
        Cv2.Flip(cannyImage, cannyImage, FlipMode.XY);
        Cv2.ImShow("SuperLaserDetect", videoSourceImage);
        Cv2.ImShow("Processed WebCam", cannyImage);
    }
    
    public void OnDestroy() {
        Cv2.DestroyAllWindows();
    }
}