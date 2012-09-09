#include  <QtCore/QCoreApplication>
#include  "opencv/cv.h"
#include  "opencv/cvaux.h"
#include  "opencv/highgui.h"
#include  <vector>
#include  <string>
#include  <algorithm>
#include  <stdio.h>
#include  <ctype.h>
#include <sys/stat.h>

#include <netinet/in.h>
#include <sys/socket.h>
#include <arpa/inet.h>
#include <stdlib.h>
#include <unistd.h>
#include <pthread.h>

using namespace std;

#define CAMERA_ID_LEFT 1
#define CAMERA_ID_RIGHT 2

#define WINDOW_RAW "Raw"
#define WINDOW_RECTIFIED "Rectified"
#define WINDOW_DEPTH "Depth"
#define WINDOW_MARKER "Marker"

#define PORT 8888

pthread_mutex_t mutex = PTHREAD_MUTEX_INITIALIZER;
pthread_mutex_t mutex2 = PTHREAD_MUTEX_INITIALIZER;

void* streamServer(void* arg);
void  quitStreamServer(const char* msg, int retval);
void* jsonServer(void* arg);
void  quitJsonServer(const char* msg, int retval);

IplImage*   sharedImage;
int         is_data_ready = 0;

int         is_json_ready = 0;
int         json_x = 0;
int         json_y = 0;
int         json_size = 0;


struct SockBundle {
    int serversock;
    int clientsock;
};
struct SockBundle streamSock;
struct SockBundle jsonSock;


struct _StereoCalibrateValues {
    double M1[3][3], M2[3][3], D1[5], D2[5];
    double R[3][3], T[3], E[3][3], F[3][3];
    CvSize imageSize;
    int imageType;
    CvMat _M1;
    CvMat _M2;
    CvMat _D1;
    CvMat _D2;
    CvMat _R;
    CvMat _T;
    CvMat _E;
    CvMat _F;
};
typedef struct _StereoCalibrateValues StereoCalibrateValues;

void doStereoCalibrate(CvCapture* leftCapture, CvCapture* rightCapture, StereoCalibrateValues& st);
void showDepthMap(CvCapture* leftCapture, CvCapture* rightCapture, StereoCalibrateValues& st);


int main(int argc, char *argv[])
{
    int width = 640;
    int height = 480;
//    int width = 352;
//    int height = 288;
    //QCoreApplication a(argc, argv);
    CvCapture* leftCapture = cvCreateCameraCapture(CAMERA_ID_LEFT);
    CvCapture* rightCapture = cvCreateCameraCapture(CAMERA_ID_RIGHT);
    printf("setting left camera width  : %d\n", cvSetCaptureProperty(leftCapture, CV_CAP_PROP_FRAME_WIDTH, width));
    printf("setting left camera height : %d\n", cvSetCaptureProperty(leftCapture, CV_CAP_PROP_FRAME_HEIGHT,height));
    printf("setting left camera width  : %d\n", cvSetCaptureProperty(rightCapture, CV_CAP_PROP_FRAME_WIDTH, width));
    printf("setting left camera height : %d\n", cvSetCaptureProperty(rightCapture, CV_CAP_PROP_FRAME_HEIGHT,height));
    fflush(stdout);

    StereoCalibrateValues st;

    {
        char fname[] = "StereoCalibrateValues.dat";
        struct stat  statValue;
        if (stat(fname, &statValue) != 0) {
            doStereoCalibrate(leftCapture, rightCapture,st);
            FILE *fp;
            fp = fopen(fname, "wb");
            fwrite(&st, sizeof(st), 1, fp);
            fclose(fp);
            fp = 0;
        } else {
            FILE *fp;
            fp = fopen(fname, "rb");
            fread(&st, sizeof(st), 1, fp);
            fclose(fp);
            fp = 0;
            st._M1 = cvMat(3, 3, CV_64F, st.M1);
            st._M2 = cvMat(3, 3, CV_64F, st.M2);
            st._D1 = cvMat(1, 5, CV_64F, st.D1);
            st._D2 = cvMat(1, 5, CV_64F, st.D2);
            st._R = cvMat(3, 3, CV_64F, st.R);
            st._T = cvMat(3, 1, CV_64F, st.T);
            st._E = cvMat(3, 3, CV_64F, st.E);
            st._F = cvMat(3, 3, CV_64F, st.F);
        }
        st.imageSize.width = width;
        st.imageSize.height = height;

        sharedImage = cvCreateImage(cvSize (st.imageSize.width, st.imageSize.height), IPL_DEPTH_8U, 1);
    }

    pthread_t   thread_s;
    pthread_t   thread_j;
    {   // サーバースレッドを開始
        if (pthread_create(&thread_s, NULL, streamServer, NULL)) {
            quitStreamServer("pthread_create failed.", 1);
        }
        if (pthread_create(&thread_j, NULL, jsonServer, NULL)) {
            quitStreamServer("pthread_create failed.", 1);
        }
    }

    {   // これがメインのループ処理
        showDepthMap(leftCapture, rightCapture, st);
    }

    {   // サーバースレッドを停止
        if (pthread_cancel(thread_s)) {
            quitStreamServer("pthread_cancel failed.", 1);
        }
        if (pthread_cancel(thread_j)) {
            quitJsonServer("pthread_cancel failed.", 1);
        }
    }

    cvReleaseCapture( &leftCapture );
    cvReleaseCapture( &rightCapture );
    cvReleaseImage(&sharedImage);
    //return a.exec();

    quitStreamServer(NULL, 0);

    return 0;
}

void doStereoCalibrate(CvCapture* leftCapture, CvCapture* rightCapture, StereoCalibrateValues& st) {
    IplImage* workImage = NULL;
    cvNamedWindow( WINDOW_RAW, CV_WINDOW_AUTOSIZE );
    int nx = 7;
    int ny = 5;
    int squareSize = 10;
    int numOfFrame = 10;
    int detectedNum = 0;
    int n = nx * ny;
    CvPoint2D32f leftPs[n*numOfFrame];
    CvPoint2D32f rightPs[n*numOfFrame];

    while (1) {
        IplImage *leftImage = cvQueryFrame(leftCapture);
        IplImage *rightImage = cvQueryFrame(rightCapture);
        if (!leftImage || !rightImage) {
            continue;
        }
        int leftResult = 0;
        int rightResult = 0;
        {
            int count = 0;
            leftResult = cvFindChessboardCorners(leftImage, cvSize(nx, ny), &leftPs[n*detectedNum],
                        &count, CV_CALIB_CB_ADAPTIVE_THRESH | CV_CALIB_CB_NORMALIZE_IMAGE);
            if (leftResult) {
                IplImage* timg = cvCreateImage(cvSize (leftImage->width, leftImage->height), 8, 1);
                cvCvtColor(leftImage, timg, CV_BGR2GRAY);
                cvFindCornerSubPix(timg, &leftPs[n*detectedNum], count, cvSize(11, 11),
                                cvSize(-1, -1),
                                cvTermCriteria(CV_TERMCRIT_ITER + CV_TERMCRIT_EPS, 30,
                                                0.01));
                cvReleaseImage(&timg);
            }
            cvDrawChessboardCorners(leftImage, cvSize(nx, ny), &leftPs[n*detectedNum], count, leftResult);
        }
        {
            int count = 0;
            rightResult = cvFindChessboardCorners(rightImage, cvSize(nx, ny), &rightPs[n*detectedNum],
                        &count, CV_CALIB_CB_ADAPTIVE_THRESH | CV_CALIB_CB_NORMALIZE_IMAGE);
            if (rightResult) {
                IplImage* timg = cvCreateImage(cvSize (rightImage->width, rightImage->height), 8, 1);
                cvCvtColor(rightImage, timg, CV_BGR2GRAY);
                cvFindCornerSubPix(timg, &rightPs[n*detectedNum], count, cvSize(11, 11),
                                cvSize(-1, -1),
                                cvTermCriteria(CV_TERMCRIT_ITER + CV_TERMCRIT_EPS, 30,
                                                0.01));
                cvReleaseImage(&timg);
            }
            cvDrawChessboardCorners(rightImage, cvSize(nx, ny), &rightPs[n*detectedNum], count, rightResult);
        }
        {
            if (workImage == NULL) {
                workImage = cvCreateImage(cvSize (leftImage->width + rightImage->width, leftImage->height), leftImage->depth, leftImage->nChannels);
            }
            CvRect roi = cvRect (0, 0, leftImage->width, leftImage->height);
            cvSetImageROI(workImage, roi);
            cvCopy(leftImage, workImage);
            roi.x = leftImage->width;
            cvSetImageROI(workImage, roi);
            cvCopy(rightImage, workImage);
            cvResetImageROI(workImage);
            cvShowImage( WINDOW_RAW, workImage );
        }
        if (leftResult && rightResult) {
            detectedNum++;
            printf("Detected:%d\n", detectedNum);
            fflush(stdout);
        }
        if (detectedNum >= numOfFrame) {
            printf("Running stereo calibration ...");
            fflush(stdout);
            {
                st.imageSize = cvGetSize(leftImage);
                cv::Mat src = cv::cvarrToMat(leftImage);
                st.imageType = src.type();
                printf("(%d,%d)\n", st.imageSize.width, st.imageSize.height);
                fflush(stdout);
            }

            vector < CvPoint3D32f > objectPoints;
            vector<int> npoints;
            {
                npoints.resize(numOfFrame, n);
                objectPoints.resize(numOfFrame * n);
                for (int i = 0; i < ny; i++) {
                    for (int j = 0; j < nx; j++) {
                        objectPoints[i * nx + j] = cvPoint3D32f(i * squareSize,
                                                j * squareSize, 0);
                    }
                }
                for (int i = 1; i < numOfFrame; i++) {
                        copy(objectPoints.begin(), objectPoints.begin() + n,
                                        objectPoints.begin() + i * n);
                }
            }

            // 配列とベクトルの格納領域
            st._M1 = cvMat(3, 3, CV_64F, st.M1);
            st._M2 = cvMat(3, 3, CV_64F, st.M2);
            st._D1 = cvMat(1, 5, CV_64F, st.D1);
            st._D2 = cvMat(1, 5, CV_64F, st.D2);
            st._R = cvMat(3, 3, CV_64F, st.R);
            st._T = cvMat(3, 1, CV_64F, st.T);
            st._E = cvMat(3, 3, CV_64F, st.E);
            st._F = cvMat(3, 3, CV_64F, st.F);

            CvMat _objectPoints = cvMat(1, n*numOfFrame, CV_32FC3, &objectPoints[0]);
            CvMat _imagePoints1 = cvMat(1, n*numOfFrame, CV_32FC2, leftPs);
            CvMat _imagePoints2 = cvMat(1, n*numOfFrame, CV_32FC2, rightPs);
            CvMat _npoints = cvMat(1, npoints.size(), CV_32S, &npoints[0]);
            cvSetIdentity(&st._M1);
            cvSetIdentity(&st._M2);
            cvZero(&st._D1);
            cvZero(&st._D2);
            // ステレオカメラをキャリブレーションする
            cvStereoCalibrate(&_objectPoints, &_imagePoints1, &_imagePoints2, &_npoints,
                            &st._M1, &st._D1, &st._M2, &st._D2, st.imageSize, &st._R, &st._T, &st._E, &st._F,
                            cvTermCriteria(CV_TERMCRIT_ITER + CV_TERMCRIT_EPS, 100, 1e-5),
                            CV_CALIB_FIX_ASPECT_RATIO + CV_CALIB_ZERO_TANGENT_DIST
                                            + CV_CALIB_SAME_FOCAL_LENGTH);
            printf("done\n");
            fflush(stdout);
            break;
        }
        char c = cvWaitKey(33);
        if( c == 27 ) break;
    }
    if (workImage) {
        cvReleaseImage(&workImage);
    }
    cvDestroyWindow( WINDOW_RAW );
}

void showDepthMap(CvCapture* leftCapture, CvCapture* rightCapture, StereoCalibrateValues& st) {
    cvNamedWindow( WINDOW_RECTIFIED, CV_WINDOW_AUTOSIZE );
    cvNamedWindow( WINDOW_DEPTH, CV_WINDOW_AUTOSIZE );
    cvNamedWindow( WINDOW_MARKER, CV_WINDOW_AUTOSIZE );

    CvMat* mx1 = cvCreateMat(st.imageSize.height, st.imageSize.width, CV_32F);
    CvMat* my1 = cvCreateMat(st.imageSize.height, st.imageSize.width, CV_32F);
    CvMat* mx2 = cvCreateMat(st.imageSize.height, st.imageSize.width, CV_32F);
    CvMat* my2 = cvCreateMat(st.imageSize.height, st.imageSize.width, CV_32F);
    CvMat* img1g = cvCreateMat(st.imageSize.height, st.imageSize.width, CV_8UC1);
    CvMat* img2g = cvCreateMat(st.imageSize.height, st.imageSize.width, CV_8UC1);
    CvMat* img1r = cvCreateMat(st.imageSize.height, st.imageSize.width, CV_8UC1);
    CvMat* img2r = cvCreateMat(st.imageSize.height, st.imageSize.width, CV_8UC1);
    CvMat* disp = cvCreateMat(st.imageSize.height, st.imageSize.width, CV_16S);
    CvMat* tempImg1 = cvCreateMat(st.imageSize.height, st.imageSize.width, CV_16S);
    CvMat* tempImg2 = cvCreateMat(st.imageSize.height, st.imageSize.width, CV_8U);
    CvMat* tempImg3 = cvCreateMat(st.imageSize.height, st.imageSize.width, CV_8U);
    //CvMat* disp2 = cvCreateMat(st.imageSize.height, st.imageSize.width, CV_16S);
    CvMat* vdisp = cvCreateMat(st.imageSize.height, st.imageSize.width, CV_8U);
    CvMat* rep = cvCreateMat(st.imageSize.height, st.imageSize.width, CV_32FC3);
    CvMat* pair = cvCreateMat(st.imageSize.height, st.imageSize.width * 2, CV_8UC1);
    double R1[3][3], R2[3][3], P1[3][4], P2[3][4], Q[4][4];
    CvMat _Q = cvMat(4, 4, CV_64F, Q);
    CvMat _R1 = cvMat(3, 3, CV_64F, R1);
    CvMat _R2 = cvMat(3, 3, CV_64F, R2);
    {   // キャリブレーションされている場合 BOUGUETの手法)(
        CvMat _P1 = cvMat(3, 4, CV_64F, P1);
        CvMat _P2 = cvMat(3, 4, CV_64F, P2);
        cvStereoRectify(&st._M1, &st._M2, &st._D1, &st._D2, st.imageSize, &st._R, &st._T, &_R1,
                        &_R2, &_P1, &_P2, &_Q, 0/*CV_CALIB_ZERO_DISPARITY*/);
        // cvRemap()用にマップをあらかじめ計算する
        cvInitUndistortRectifyMap(&st._M1, &st._D1, &_R1, &_P1, mx1, my1);
        cvInitUndistortRectifyMap(&st._M2, &st._D2, &_R2, &_P2, mx2, my2);
    }

    int Zmin= -100.0;
    int Zmax= -50.0;
    //CvStereoGCState *GCState = cvCreateStereoGCState(16, 2);

    CvStereoBMState *BMState = cvCreateStereoBMState();
    assert(BMState != 0);
    BMState->preFilterSize = 31;
    BMState->preFilterCap = 31;
    BMState->SADWindowSize = 31;
    BMState->minDisparity = -16;
    BMState->numberOfDisparities = 64;
    BMState->textureThreshold = 10;
    BMState->uniquenessRatio = 15;
    IplConvKernel* kernel = cvCreateStructuringElementEx(5,5,3,3,CV_SHAPE_ELLIPSE, NULL);

    while (1) {
        IplImage *leftImage=cvQueryFrame(leftCapture);
        IplImage *rightImage=cvQueryFrame(rightCapture);
        if (!leftImage || !rightImage) {
            continue;
        }
        {
            cvCvtColor(leftImage, img1g, CV_BGR2GRAY);
            cvCvtColor(rightImage, img2g, CV_BGR2GRAY);
        }
        {
            CvMat part;
            cvRemap(img1g, img1r, mx1, my1);
            cvRemap(img2g, img2r, mx2, my2);
            {
                cvGetCols(pair, &part, 0, st.imageSize.width);
                cvCopy(img1r, &part);
                cvGetCols(pair, &part, st.imageSize.width,
                                st.imageSize.width * 2);
                cvCopy(img2r, &part);
                for (int j = 0; j < st.imageSize.height; j += 16) {
                        cvLine(pair, cvPoint(0, j),
                                        cvPoint(st.imageSize.width * 2, j),
                                        CV_RGB(0, 255, 0));
                }
                cvShowImage( WINDOW_RECTIFIED, pair );
            }
            {

                // cvFindStereoCorrespondenceGC(img1r,img2r, disp, disp2, GCState, 0);
                cvFindStereoCorrespondenceBM(img1r, img2r, disp, BMState);
                if (0) {
                    cvNormalize(disp, vdisp, 0, 256, CV_MINMAX);
                    cvShowImage(WINDOW_DEPTH, vdisp);
                } else {
                    {   // Reprojectを行う
                        cvReprojectImageTo3D(disp, rep, &_Q);
                        float* fptr= rep->data.fl;
                        float fv;
                        double scaleit = 255.0 / (Zmax - Zmin);
                        unsigned char *cptr = vdisp->data.ptr;
                        fptr = rep->data.fl;
                        //LOOP
                        for(int y=0; y<st.imageSize.height; ++y)
                        {
                            for(int x=0; x<st.imageSize.width; ++x)
                            {
                                fv = *(fptr+2);  // read the depth
                                if((fv >= 0.0)||(fv > Zmax)) // too close
                                    *cptr++ = 0;
                                else if(fv < Zmin)    // too far
                                    *cptr++ = 0;
                                else
                                    *cptr++ = (unsigned char)( scaleit*( (double)fv - Zmin) );
                                fptr+= 3;
                            }
                        }
                        cvMorphologyEx(vdisp, vdisp, tempImg1, kernel, CV_MOP_CLOSE, 1);
                    }
                    {   // 手を検出させる
                        CvMoments moments;
                        cvThreshold(vdisp, tempImg2, 64, 255, CV_THRESH_BINARY);
                        cvMoments(tempImg2, &moments, 0);
                        double m00 = cvGetSpatialMoment(&moments, 0, 0);
                        double m10 = cvGetSpatialMoment(&moments, 1, 0);
                        double m01 = cvGetSpatialMoment(&moments, 0, 1);
                        int gX = m10/m00;
                        int gY = m01/m00;

                        int r;
                        int space = 0;
                        { // 面積の計算
                            for(int y = 0; y < tempImg2->height; y++) {
                                for(int x = 0; x < tempImg2->width; x++) {
                                    const int val = tempImg2->data.ptr[y * tempImg2->cols + x];
                                    if(val & 1) {
                                        ++space;//白色なのでカウントする
                                    }
                                }
                            }
                            r = sqrt(space);
                        }

                        cvCircle(vdisp, cvPoint(gX, gY), r, CV_RGB(255,255,255), 6, 8, 0);
                        {
                            pthread_mutex_lock(&mutex2);
                            cvCopy(img1g, sharedImage);
                            cvCircle(sharedImage, cvPoint(gX, gY), r, CV_RGB(255,255,255), 6, 8, 0);
                            is_data_ready = 1;
                            json_x = gX;
                            json_y = gY;
                            json_size = r;
                            is_json_ready = 1;
                            pthread_mutex_unlock(&mutex2);
                        }
                    }

                    cvShowImage( WINDOW_MARKER, tempImg2);
                    cvShowImage( WINDOW_DEPTH, vdisp );
                }
            }
        }

        char c = cvWaitKey(33);
        if( c == 27 ) break;
    }
    cvReleaseMat(&vdisp);
    cvReleaseMat(&disp);
    cvReleaseMat(&tempImg1);
    cvReleaseMat(&tempImg2);
    cvReleaseMat(&tempImg3);
    cvReleaseStructuringElement(&kernel);
    cvReleaseMat(&pair);
    cvReleaseMat(&rep);
//    cvReleaseStereoGCState(&GCState);
    cvReleaseStereoBMState(&BMState);
    cvDestroyWindow( WINDOW_MARKER );
    cvDestroyWindow( WINDOW_DEPTH );
    cvDestroyWindow( WINDOW_RECTIFIED );
}

/**
 * This is the streaming server, run as a separate thread
 * This function waits for a client to connect, and send the grayscaled images
 */
void* streamServer(void* arg)
{
    struct sockaddr_in server;

    /* make this thread cancellable using pthread_cancel() */
    pthread_setcancelstate(PTHREAD_CANCEL_ENABLE, NULL);
    pthread_setcanceltype(PTHREAD_CANCEL_ASYNCHRONOUS, NULL);

    /* open socket */
    if ((streamSock.serversock = socket(PF_INET, SOCK_STREAM, 0)) == -1) {
        quitStreamServer("socket() failed", 1);
    }

    /* setup server's IP and port */
    memset(&server, 0, sizeof(server));
    server.sin_family = AF_INET;
    server.sin_port = htons(PORT);
    server.sin_addr.s_addr = INADDR_ANY;

    /* bind the socket */
    if (bind(streamSock.serversock, (const sockaddr*)&server, sizeof(server)) == -1) {
        quitStreamServer("bind() failed", 1);
    }

    /* wait for connection */
    if (listen(streamSock.serversock, 10) == -1) {
        quitStreamServer("listen() failed.", 1);
    }

    /* accept a client */
    if ((streamSock.clientsock = accept(streamSock.serversock, NULL, NULL)) == -1) {
        quitStreamServer("accept() failed", 1);
    }

    /* the size of the data to be sent */
    int imgsize = sharedImage->imageSize;
    int bytes;
    {   // output width and height
        uint16_t w = 0xFFFF & sharedImage->width;
        uint16_t h = 0xFFFF & sharedImage->height;
        bytes = send(streamSock.clientsock, &w, 2, 0);
        bytes = send(streamSock.clientsock, &h, 2, 0);
    }

    /* start sending images */
    while(1)
    {
        /* send the grayscaled frame, thread safe */
        if (is_data_ready) {
            pthread_mutex_lock(&mutex);
            bytes = send(streamSock.clientsock, sharedImage->imageData, imgsize, 0);
            is_data_ready = 0;
            pthread_mutex_unlock(&mutex);
        }

        /* if something went wrong, restart the connection */
        if (bytes != imgsize) {
            fprintf(stderr, "Connection closed.\n");
            close(streamSock.clientsock);

            if ((streamSock.clientsock = accept(streamSock.serversock, NULL, NULL)) == -1) {
                quitStreamServer("accept() failed", 1);
            }
            {   // output width and height
                uint16_t w = 0xFFFF & sharedImage->width;
                uint16_t h = 0xFFFF & sharedImage->height;
                bytes = send(streamSock.clientsock, &w, 2, 0);
                bytes = send(streamSock.clientsock, &h, 2, 0);
            }
        }

        /* have we terminated yet? */
        pthread_testcancel();

        /* no, take a rest for a while */
        usleep(1000);
    }

    return NULL;
}

void quitStreamServer(const char* msg, int retval)
{
    if (retval == 0) {
        fprintf(stdout, (msg == NULL ? "" : msg));
        fprintf(stdout, "\n");
    } else {
        fprintf(stderr, (msg == NULL ? "" : msg));
        fprintf(stderr, "\n");
    }

    if (streamSock.clientsock) close(streamSock.clientsock);
    if (streamSock.serversock) close(streamSock.serversock);

    pthread_mutex_destroy(&mutex);

    exit(retval);
}

void* jsonServer(void* arg)
{
    struct sockaddr_in server;

    /* make this thread cancellable using pthread_cancel() */
    pthread_setcancelstate(PTHREAD_CANCEL_ENABLE, NULL);
    pthread_setcanceltype(PTHREAD_CANCEL_ASYNCHRONOUS, NULL);

    /* open socket */
    if ((jsonSock.serversock = socket(PF_INET, SOCK_STREAM, 0)) == -1) {
        quitJsonServer("socket() failed", 1);
    }

    /* setup server's IP and port */
    memset(&server, 0, sizeof(server));
    server.sin_family = AF_INET;
    server.sin_port = htons(PORT+1);
    server.sin_addr.s_addr = INADDR_ANY;

    /* bind the socket */
    if (bind(jsonSock.serversock, (const sockaddr*)&server, sizeof(server)) == -1) {
        quitJsonServer("bind() failed", 1);
    }

    /* wait for connection */
    if (listen(jsonSock.serversock, 10) == -1) {
        quitJsonServer("listen() failed.", 1);
    }

    /* accept a client */
    if ((jsonSock.clientsock = accept(jsonSock.serversock, NULL, NULL)) == -1) {
        quitJsonServer("accept() failed", 1);
    }

    /* the size of the data to be sent */
    int bytes;
    int id;

    {
        char b = 0;
        bool flag = false;
        // TODO ここにヘッダを読み込むまで待つ
        while (recv(jsonSock.clientsock, &b, 1, 0) > 0) {
            if (flag) {
                if (b == '\n') {
                    break;
                } else if (b == '\r') {
                    // ignore
                } else {
                    flag = false;
                }
            } else {
                if (b == '\n') {
                    flag = true;
                } else if (b == '\r') {
                    // ignore
                } else {
                    flag = false;
                }
            }
        }
    }
    {   // send header
        const char *header = "HTTP/1.1 200\nAccess-Control-Allow-Credentials: true\nContent-Type: text/event-stream\nAccess-Control-Allow-Origin:*\n\n";
        bytes = send(jsonSock.clientsock, header, strlen(header), 0);
    }

    /* start sending images */
    char buf[1024];
    while(1)
    {
        /* send the grayscaled frame, thread safe */
        if (is_json_ready) {
            pthread_mutex_lock(&mutex2);
            int length = sprintf(buf, "id: %d\ndata: { \"x\": %d, \"y\": %d, \"size\": %d }\n\n", id++, json_x, json_y, json_size);
            bytes = send(jsonSock.clientsock, buf, length, 0);
            is_json_ready = 0;
            pthread_mutex_unlock(&mutex2);
        }

        /* if something went wrong, restart the connection */
        if (!bytes) {
            fprintf(stderr, "Connection closed.\n");
            close(jsonSock.clientsock);

            if ((jsonSock.clientsock = accept(jsonSock.serversock, NULL, NULL)) == -1) {
                quitJsonServer("accept() failed", 1);
            }
        }

        /* have we terminated yet? */
        pthread_testcancel();

        /* no, take a rest for a while */
        usleep(1000);
    }

    return NULL;
}
void quitJsonServer(const char* msg, int retval)
{
    if (retval == 0) {
        fprintf(stdout, (msg == NULL ? "" : msg));
        fprintf(stdout, "\n");
    } else {
        fprintf(stderr, (msg == NULL ? "" : msg));
        fprintf(stderr, "\n");
    }

    if (jsonSock.clientsock) close(jsonSock.clientsock);
    if (jsonSock.serversock) close(jsonSock.serversock);

    pthread_mutex_destroy(&mutex2);

    exit(retval);
}
