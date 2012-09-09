#-------------------------------------------------
#
# Project created by QtCreator 2012-05-26T11:18:44
#
#-------------------------------------------------

QT       -= gui

TARGET = OpenCvStereo
CONFIG   += console
CONFIG   -= app_bundle

TEMPLATE = app

SOURCES += main.cpp

INCLUDEPATH += /home/cattaka/opt/OpenCV-2.3.1/include
LIBS        += -L/home/cattaka/opt/OpenCV-2.3.1/lib
LIBS        += -lopencv_core -lopencv_highgui -lopencv_calib3d
