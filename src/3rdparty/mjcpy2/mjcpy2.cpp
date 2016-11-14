#include <boost/numpy.hpp>
#include <cmath>
#include "macros.h"
#include <iostream>
#include <boost/python/slice.hpp>
#include "mujoco_osg_viewer.hpp"


#include "mujoco.h"
#include "glfw3.h"
#include "stdlib.h"
#include "string.h"
#include <mutex>
#include <thread>

namespace bp = boost::python;
namespace bn = boost::numpy;


// include pressure dynamics 
//#include "mj_prDyn.cpp"

namespace {

bp::object main_namespace;

template<typename T>
bn::ndarray toNdarray1(const T* data, long dim0) {
  long dims[1] = {dim0};
  bn::ndarray out = bn::empty(1, dims, bn::dtype::get_builtin<T>());
  memcpy(out.get_data(), data, dim0*sizeof(T));
  return out;
}
template<typename T>
bn::ndarray toNdarray2(const T* data, long dim0, long dim1) {
  long dims[2] = {dim0,dim1};
  bn::ndarray out = bn::empty(2, dims, bn::dtype::get_builtin<T>());
  memcpy(out.get_data(), data, dim0*dim1*sizeof(T));
  return out;
}
template<typename T>
bn::ndarray toNdarray3(const T* data, long dim0, long dim1, long dim2) {
  long dims[3] = {dim0,dim1,dim2};
  bn::ndarray out = bn::empty(3, dims, bn::dtype::get_builtin<T>());
  memcpy(out.get_data(), data, dim0*dim1*dim2*sizeof(T));
  return out;
}


bool endswith(const std::string& fullString, const std::string& ending)
{
	return (fullString.length() >= ending.length()) && 
		(0 == fullString.compare(fullString.length() - ending.length(), ending.length(), ending));
}

    class StateBase
    {
    public:
        virtual void keyboard(GLFWwindow *window, int key, int scancode, int act, int mods) = 0; /* purely abstract function */
        virtual void mouse_button(GLFWwindow* window, int button, int act, int mods) = 0; /* purely abstract function */
        virtual void mouse_move(GLFWwindow* window, double xpos, double ypos) = 0; /* purely abstract function */
        virtual void scroll(GLFWwindow* window, double xoffset, double yoffset) = 0; /* purely abstract function */
        virtual void drop(GLFWwindow* window, int count, const char** paths) = 0; /* purely abstract function */
        
        static StateBase *keyboard_event_handling_instance;
        virtual void setkeyboardEventHandling() { keyboard_event_handling_instance = this; }
        
        static void keyboard_dispatch(GLFWwindow *window, int key, int scancode, int act, int mods)
        {
            if(keyboard_event_handling_instance)
                keyboard_event_handling_instance->keyboard(window,key,scancode,act,mods);
        }
        
        
        static StateBase *event_handling_instance;
        // technically setEventHandling should be finalized so that it doesn't
        // get overwritten by a descendant class.
        virtual void setEventHandling() { event_handling_instance = this; }
        
        static void mouse_button_dispatch(GLFWwindow* window, int button, int act, int mods)
        {
            if(event_handling_instance)
                event_handling_instance->mouse_button(window,button,act,mods);
        }
        
        
        static StateBase *mouse_move_event_handling_instance;
        virtual void setMouseMoveEventHandling() { mouse_move_event_handling_instance = this; }
        
        static void mouse_move_dispatch(GLFWwindow* window, double xpos, double ypos)
        {
            if(mouse_move_event_handling_instance)
                mouse_move_event_handling_instance->mouse_move(window, xpos, ypos);
        }
        
        
        
        static StateBase *scroll_event_handling_instance;
        // technically setEventHandling should be finalized so that it doesn't
        // get overwritten by a descendant class.
        virtual void setScrollEventHandling() { scroll_event_handling_instance = this; }
        
        static void scroll_dispatch(GLFWwindow* window, double xoffset, double yoffset)
        {
            if(scroll_event_handling_instance)
                scroll_event_handling_instance->scroll(window, xoffset, yoffset);
        }
        
        static StateBase *drop_event_handling_instance;
        // technically setEventHandling should be finalized so that it doesn't
        // get overwritten by a descendant class.
        virtual void setDropEventHandling() { drop_event_handling_instance = this; }
        
        static void drop_dispatch(GLFWwindow* window, int count, const char** paths)
        {
            if(drop_event_handling_instance)
                drop_event_handling_instance->drop(window, count, paths);
        }
        
        
    };
    

    // funny thing that you have to omit `static` here. Learn about global scope
    // type qualifiers to understand why.
    StateBase * StateBase::event_handling_instance;
    StateBase * StateBase::keyboard_event_handling_instance;
    StateBase * StateBase::mouse_move_event_handling_instance;
    StateBase * StateBase::scroll_event_handling_instance;
    StateBase * StateBase::drop_event_handling_instance;


class PyMJCWorld2 : StateBase {


public:

    PyMJCWorld2(const std::string& loadfile);
    bp::object Step(const bn::ndarray& x, const bn::ndarray& u);
    void Plot(const bn::ndarray& x);
    void InitCam(float cx,float cy,float cz,float px,float py,float pz);
    void InitViewer(int width, int height, float cx,float cy,float cz,float px,float py,float pz);
    void Idle(const bn::ndarray& x);
    bn::ndarray GetCOMMulti(const bn::ndarray& x);
    bn::ndarray GetJacSite(int site);
    void Kinematics();
    bp::dict GetModel();
    void SetModel(bp::dict d);
    bp::dict GetData();
    void SetData(bp::dict d);
    bp::dict GetImage();
    bp::dict GetImageScaled(int width, int height);
    void SetNumSteps(int n) {m_numSteps=n;}
    void SetCamera(float x, float y, float z, float px, float py, float pz);
    int visualize_render(const std::string& loadfile);
    void loadmodel(GLFWwindow* window, const char* filename, const char* xmlstring);
    void autoscale(GLFWwindow* window);
    virtual void keyboard(GLFWwindow* window, int key, int scancode, int act, int mods);
    virtual void mouse_button(GLFWwindow* window, int button, int act, int mods);
    virtual void mouse_move(GLFWwindow* window, double xpos, double ypos);
    virtual void scroll(GLFWwindow* window, double xoffset, double yoffset);
    virtual void drop(GLFWwindow* window, int count, const char** paths);
    void makeoptionstring(const char* name, char key, char* buf);
    void advance(void);
    void render(GLFWwindow* window);
    double min_act(int act_index);
    double max_act(int act_index);
    ~PyMJCWorld2();
private:
//
//    void _PlotInit();
//    void _PlotInit(float x, float y, float z, float px, float py, float pz);
//    void _PlotInit(int width, int height, float x, float y, float z, float px, float py, float pz);

    mjModel* m_model;
    mjData* m_data;
    mjData mj_data_plot;
    MujocoOSGViewer* m_viewer;
    int m_numSteps;
    int m_featmask;
    
    // synchronization
    std::mutex gui_mutex;
    
    // model
    mjModel* m;
    mjData* d;
    
    char lastfile[1000];
    
    // user state
    bool paused;
    bool showoption;
    bool showinfo;
    bool showdepth;
    int showhelp;                   // 0: none; 1: brief; 2: full
    int speedtype;                  // 0: slow; 1: normal; 2: max
    
    // abstract visualization
    mjvObjects objects;
    mjvCamera cam;
    mjvOption vopt;
    char status[1000];
    
    // OpenGL rendering
    mjrContext con;
    mjrOption ropt;
    double scale;
    bool stereoavailable;
    float depth_buffer[5120*2880];        // big enough for 5K screen
    unsigned char depth_rgb[1280*720*3];  // 1/4th of screen
    
    // selection and perturbation
    bool button_left;
    bool button_middle;
    bool button_right;
    int lastx;
    int lasty;
    int selbody;
    int perturb;
    mjtNum selpos[3];
    mjtNum refpos[3];
    mjtNum refquat[4];
    int needselect;                 // 0: none, 1: select, 2: center
    
    char opt_title[1000];
    char opt_content[1000];
    GLFWwindow* window;
    
    std::thread rendering_th;
    bool isVizualizer;

};

    

int PyMJCWorld2::visualize_render(const std::string& loadfile)
{
    printf("Launching visualizer\n");
    window = 0;
    // init GLFW, set multisampling
    if (!glfwInit())
        return 1;
    
    
    glfwWindowHint(GLFW_SAMPLES, 4);
    
    // try stereo if refresh rate is at least 100Hz
    if( glfwGetVideoMode(glfwGetPrimaryMonitor())->refreshRate>=100 )
    {
        glfwWindowHint(GLFW_STEREO, 1);
        window = glfwCreateWindow(1024, 768, "Visualize", NULL, NULL);
        isVizualizer = true;
        if( window )
            stereoavailable = true;
    }
    
    // no stereo: try mono
    if( !window )
    {
        glfwWindowHint(GLFW_STEREO, 0);
        window = glfwCreateWindow(600, 400, "Visualize", NULL, NULL);
        isVizualizer = true;
    }
    if( !window )
    {
        glfwTerminate();
        return 1;
    }
    glfwMakeContextCurrent(window);
    
    // determine retina scaling
    int width, width1, height;
    glfwGetFramebufferSize(window, &width, &height);
    glfwGetWindowSize(window, &width1, &height);
    scale = (double)width/(double)width1;
    
    // init MuJoCo rendering
    mjv_makeObjects(&objects, 1000);
    mjv_defaultCamera(&cam);
    mjv_defaultOption(&vopt);
    mjr_defaultOption(&ropt);
    mjr_defaultContext(&con);
    mjr_makeContext(m, &con, 150); //??? creating context with empty model!
    printf("LOADING MODEL\n");
    loadmodel(window, loadfile.c_str(), 0);
    

    // set GLFW callbacks
    glfwSetKeyCallback(window, StateBase::keyboard_dispatch);
    glfwSetCursorPosCallback(window, StateBase::mouse_move_dispatch);
    glfwSetMouseButtonCallback(window, StateBase::mouse_button_dispatch);
    glfwSetScrollCallback(window, StateBase::scroll_dispatch);
    glfwSetDropCallback(window, StateBase::drop_dispatch);
    

    // register call backs
//    mjcb_act_dyn = (mjfAct)pressureLocal;

    // Prepare window:: Set title + autoscale
    if (window && m->names)
        glfwSetWindowTitle(window, m->names);
    if (m)
        autoscale(window);
    
    // Prepare window:: Set title + autoscale
    if (window && m->names)
        glfwSetWindowTitle(window, m->names);
    if (m)
        autoscale(window);

    // main loop
    double last_frame_time = glfwGetTime();
    double max_fps = 100.0;
    while (isVizualizer)
    {
        // render
        render(window);
        
        // finalize
        glfwSwapBuffers(window);
        glfwPollEvents();
        
        // yield
        while( glfwGetTime()-last_frame_time < 1/max_fps )    
            std::this_thread::yield();

        // get current time
        last_frame_time = glfwGetTime();
        
        // perturb ??
        // mjv_mousePerturb(model, data, selbody, perturb, refpos, refquat, data->xfrc_applied+6*selbody);
        
        // check if we should close
        if(glfwWindowShouldClose(window))
            isVizualizer = false;
    }
    
    mjr_freeContext(&con);
    mjv_freeObjects(&objects);
    glfwTerminate();
    printf("Visualization exiting\n");
    return 0;



}
  

PyMJCWorld2::PyMJCWorld2(const std::string& loadfile) {

    mj_activate("src/3rdparty/mjpro/mjkey.txt");

    // fire visualizer
    m_viewer = NULL;
    m_numSteps = 1;
    m_featmask = 0;
    
    // user state
    paused = false;
    showoption = false;
    showinfo = true;
    showdepth = false;
    showhelp = 0;                   // 0: none; 1: brief; 2: full
    speedtype = 1;                  // 0: slow; 1: normal; 2: max
    
    scale = 1;
    stereoavailable = false;
    
    // selection and perturbation
    button_left = false;
    button_middle = false;
    button_right =  false;
    lastx = 0;
    lasty = 0;
    selbody = 0;
    perturb = 0;
    selpos[0] = 0;
    selpos[1] = 0;
    selpos[2] = 0;
    
    refpos[0] = 0;
    refpos[1] = 0;
    refpos[2] = 0;
    
    refquat[0] = 1;
    refquat[1] = 0;
    refquat[2] = 0;
    refquat[3] = 0;
    needselect = 0;                 // 0: none, 1: select, 2: center


    m = 0;
    d = 0;
    rendering_th = std::thread(&PyMJCWorld2::visualize_render, this, loadfile);
    

    // Wait for Model
    bool isModel = false;
    for(int i=0; i<100; i++)
    {
        printf("Waiting for model...\n");
        if(m_model)
        {   printf("Model Loaded\n");
            break;
        }
        else
            usleep(1000000);
    }

    if (!m_model) 
        PRINT_AND_THROW("couldn't load model: " + std::string(loadfile));
    FAIL_IF_FALSE(!!m_data);
    
    this->setEventHandling();
    this->setkeyboardEventHandling();
    this->setMouseMoveEventHandling();
    this->setScrollEventHandling();
    this->setDropEventHandling();
}


PyMJCWorld2::~PyMJCWorld2() {

    // Close visualization
    isVizualizer = false;
    if(rendering_th.joinable())
        rendering_th.join();

    // Clear model and data
	mj_deleteData(m_data);
	mj_deleteModel(m_model);
}
    
    
    void PyMJCWorld2::autoscale(GLFWwindow* window)
    {
        // autoscale
        cam.lookat[0] = m->stat.center[0];
        cam.lookat[1] = m->stat.center[1];
        cam.lookat[2] = m->stat.center[2];
        cam.distance = 1.5 * m->stat.extent;
        cam.camid = -1;
        cam.trackbodyid = -1;
        if( window )
        {
            int width, height;
            glfwGetFramebufferSize(window, &width, &height);
            mjv_updateCameraPose(&cam, (mjtNum)width/(mjtNum)height);
        }
    }
    
    
    
    // load mjb or xml model
    void PyMJCWorld2::loadmodel(GLFWwindow* window, const char* filename, const char* xmlstring)
    {
        // make sure one source is given
        if( !filename && !xmlstring )
            return;
        
        // load and compile
        char error[1000] = "could not load binary model";
        mjModel* mnew = 0;
        if( xmlstring )
            mnew = mj_loadXML(0, xmlstring, error, 1000);
        else if( strlen(filename)>4 && !strcmp(filename+strlen(filename)-4, ".mjb") )
            mnew = mj_loadModel(filename, 0, 0);
        else
            mnew = mj_loadXML(filename, 0, error, 1000);
        if( !mnew )
        {
            printf("%s\n", error);
            return;
        }
        
        // delete old model, assign new
        mj_deleteData(d);
        mj_deleteModel(m);
        m = mnew;
        d = mj_makeData(m);
        mj_forward(m, d);
        
        // save filename for reload
        if( !xmlstring )
            strcpy(lastfile, filename);
        else
            lastfile[0] = 0;
        
        // re-create custom context
        mjr_makeContext(m, &con, 150);
        
        // clear perturbation state
        perturb = 0;
        selbody = 0;
        needselect = 0;
        
        // set title
        if( window && m->names )
            glfwSetWindowTitle(window, m->names);
        
        // center and scale view
        autoscale(window);
        // will this be enough to copy by reference?
        //need to check this
        m_model = m;
        m_data = d;
        //mj_saveXML("model.xml",m, nullptr, 0);
        mj_data_plot = *mj_makeData(m);
    }

    
    // keyboard
    void PyMJCWorld2::keyboard(GLFWwindow* window, int key, int scancode, int act, int mods)
    {
        int n;
        
        // require model
        if( !m )
            return;
        
        // do not act on release
        if( act==GLFW_RELEASE )
            return;
        
        gui_mutex.lock();
        
        switch( key )
        {
            case GLFW_KEY_F1:                   // help
                showhelp++;
                if( showhelp>2 )
                    showhelp = 0;
                break;
                
            case GLFW_KEY_F2:                   // option
                showoption = !showoption;
                break;
                
            case GLFW_KEY_F3:                   // info
                showinfo = !showinfo;
                break;
                
            case GLFW_KEY_F4:                   // depthmap
                showdepth = !showdepth;
                break;
                
            case GLFW_KEY_F5:                   // stereo
                if( stereoavailable )
                    ropt.stereo = !ropt.stereo;
                break;
                
            case GLFW_KEY_F6:                   // cycle over frame rendering modes
                vopt.frame = (vopt.frame+1) % mjNFRAME;
                break;
                
            case GLFW_KEY_F7:                   // cycle over labeling modes
                vopt.label = (vopt.label+1) % mjNLABEL;
                break;
                
            case GLFW_KEY_ENTER:                // speed
                speedtype += 1;
                if( speedtype>2 )
                    speedtype = 0;
                break;
                
            case GLFW_KEY_SPACE:                // pause
                paused = !paused;
                break;
                
            case GLFW_KEY_BACKSPACE:            // reset
                mj_resetData(m, d);
                mj_forward(m, d);
                break;
                
            case GLFW_KEY_RIGHT:                // step forward
                if( paused )
                    mj_step(m, d);
                break;
                
            case GLFW_KEY_LEFT:                 // step back
                if( paused )
                {
                    m->opt.timestep = -m->opt.timestep;
                    mj_step(m, d);
                    m->opt.timestep = -m->opt.timestep;
                }
                break;
                
            case GLFW_KEY_PAGE_DOWN:            // step forward 100
                if( paused )
                    for( n=0; n<100; n++ )
                        mj_step(m,d);
                break;
                
            case GLFW_KEY_PAGE_UP:              // step back 100
                if( paused )
                {
                    m->opt.timestep = -m->opt.timestep;
                    for( n=0; n<100; n++ )
                        mj_step(m,d);
                    m->opt.timestep = -m->opt.timestep;
                }
                break;
                
            case GLFW_KEY_LEFT_BRACKET:         // previous camera
                if( cam.camid>-1 )
                    cam.camid--;
                break;
                
            case GLFW_KEY_RIGHT_BRACKET:        // next camera
                if( cam.camid<m->ncam-1 )
                    cam.camid++;
                break;
                
            default:
                // control keys
                if( mods & GLFW_MOD_CONTROL )
                {
                    if( key==GLFW_KEY_A )
                        autoscale(window);
                    else if( key==GLFW_KEY_L && lastfile[0] )
                        loadmodel(window, lastfile, 0);
                    
                    break;
                }
                
                // toggle visualization flag
                for( int i=0; i<mjNVISFLAG; i++ )
                    if( key==mjVISSTRING[i][2][0] )
                        vopt.flags[i] = !vopt.flags[i];
                
                // toggle rendering flag
                for( int i=0; i<mjNRNDFLAG; i++ )
                    if( key==mjRNDSTRING[i][2][0] )
                        ropt.flags[i] = !ropt.flags[i];
                
                // toggle geom/site group
                for( int i=0; i<mjNGROUP; i++ )
                    if( key==i+'0')
                    {
                        if( mods & GLFW_MOD_SHIFT )
                            vopt.sitegroup[i] = !vopt.sitegroup[i];
                        else
                            vopt.geomgroup[i] = !vopt.geomgroup[i];
                    }
        }
        
        gui_mutex.unlock();
    }
//
//    
//    // mouse button
    void PyMJCWorld2::mouse_button(GLFWwindow* window, int button, int act, int mods)
    {
        // past data for double-click detection
        static int lastbutton = 0;
        static double lastclicktm = 0;
        
        // update button state
        button_left =   (glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_LEFT)==GLFW_PRESS);
        button_middle = (glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_MIDDLE)==GLFW_PRESS);
        button_right =  (glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_RIGHT)==GLFW_PRESS);
        
        // update mouse position
        double x, y;
        glfwGetCursorPos(window, &x, &y);
        lastx = (int)(scale*x);
        lasty = (int)(scale*y);
        
        // require model
        if( !m )
            return;
        
        gui_mutex.lock();
        
        // set perturbation
        int newperturb = 0;
        if( (mods & GLFW_MOD_CONTROL) && selbody>0 )
        {
            // right: translate;  left: rotate
            if( button_right )
                newperturb = mjPERT_TRANSLATE;
            else if( button_left )
                newperturb = mjPERT_ROTATE;
            
            // perturbation onset: reset reference
            if( newperturb && !perturb )
            {
                int id = paused ? m->body_rootid[selbody] : selbody;
                mju_copy3(refpos, d->xpos+3*id);
                mju_copy(refquat, d->xquat+4*id, 4);
            }
        }
        perturb = newperturb;
        
        // detect double-click (250 msec)
        if( act==GLFW_PRESS && glfwGetTime()-lastclicktm<0.25 && button==lastbutton )
        {
            if( button==GLFW_MOUSE_BUTTON_LEFT )
                needselect = 1;
            else
                needselect = 2;
            
            // stop perturbation on select
            perturb = 0;
        }
        
        // save info
        if( act==GLFW_PRESS )
        {
            lastbutton = button;
            lastclicktm = glfwGetTime();
        }
        
        gui_mutex.unlock();
    }
//
//    
//    
//    // mouse move
    void PyMJCWorld2::mouse_move(GLFWwindow* window, double xpos, double ypos)
    {
        // no buttons down: nothing to do
        if( !button_left && !button_middle && !button_right )
            return;
        
        // compute mouse displacement, save
        float dx = (int)(scale*xpos) - (float)lastx;
        float dy = (int)(scale*ypos) - (float)lasty;
        lastx = (int)(scale*xpos);
        lasty = (int)(scale*ypos);
        
        // require model
        if( !m )
            return;
        
        // get current window size
        int width, height;
        glfwGetFramebufferSize(window, &width, &height);
        
        // get shift key state
        bool mod_shift = (glfwGetKey(window, GLFW_KEY_LEFT_SHIFT)==GLFW_PRESS ||
                          glfwGetKey(window, GLFW_KEY_RIGHT_SHIFT)==GLFW_PRESS);
        
        // determine action based on mouse button
        mjtMouse action;
        if( button_right )
            action = mod_shift ? mjMOUSE_MOVE_H : mjMOUSE_MOVE_V;
        else if( button_left )
            action = mod_shift ? mjMOUSE_ROTATE_H : mjMOUSE_ROTATE_V;
        else
            action = mjMOUSE_ZOOM;
        
        gui_mutex.lock();
        
        // perturbation
        if( perturb )
        {
            if( selbody>0 )
                mjv_moveObject(action, dx, dy, &cam.pose,
                               (float)width, (float)height, refpos, refquat);
        }
        
        // camera control
        else
            mjv_moveCamera(action, dx, dy, &cam, (float)width, (float)height);
        
        gui_mutex.unlock();
    }
    
    
    // scroll
    void PyMJCWorld2::scroll(GLFWwindow* window, double xoffset, double yoffset)
    {
        // require model
        if( !m )
            return;
        
        // get current window size
        int width, height;
        glfwGetFramebufferSize(window, &width, &height);
        
        // scroll
        gui_mutex.lock();
        mjv_moveCamera(mjMOUSE_ZOOM, 0, (float)(-20*yoffset), &cam, (float)width, (float)height);
        gui_mutex.unlock();
    }
    
    
    // drop
    void PyMJCWorld2::drop(GLFWwindow* window, int count, const char** paths)
    {
        // make sure list is non-empty
        if( count>0 )
        {
            gui_mutex.lock();
            loadmodel(window, paths[0], 0);
            gui_mutex.unlock();
        }
    }
    
    
    
    // make option string
    void PyMJCWorld2::makeoptionstring(const char* name, char key, char* buf)
    {
        int i=0, cnt=0;
        
        // copy non-& characters
        while( name[i] && i<50 )
        {
            if( name[i]!='&' )
                buf[cnt++] = name[i];
            
            i++;
        }
        
        // finish
        buf[cnt] = ' ';
        buf[cnt+1] = '(';
        buf[cnt+2] = key;
        buf[cnt+3] = ')';
        buf[cnt+4] = 0;
    }
    
    
    // advance simulation
    void PyMJCWorld2::advance(void)
    {
        // perturbations
        if( selbody>0 )
        {
            // fixed object: edit
            if( m->body_jntnum[selbody]==0 && m->body_parentid[selbody]==0 )
                mjv_mouseEdit(m, d, selbody, perturb, refpos, refquat);
            
            // movable object: set mouse perturbation
            else
                mjv_mousePerturb(m, d, selbody, perturb, refpos, refquat,
                                 d->xfrc_applied+6*selbody);
        }
        
        // advance simulation
        mj_step(m, d);
        
        // clear perturbation
        if( selbody>0 )
            mju_zero(d->xfrc_applied+6*selbody, 6);
    }
    
    
    // render
    void PyMJCWorld2::render(GLFWwindow* window)
    {
        // past data for FPS calculation
        static double lastrendertm = 0;
        
        // get current window rectangle
        mjrRect rect = {0, 0, 0, 0};
        glfwGetFramebufferSize(window, &rect.width, &rect.height);
        
        double duration = 0;
        gui_mutex.lock();
        
        // start timers
        double starttm = glfwGetTime();
        mjtNum startsimtm = mj_data_plot.time;
        
        
        // update simulation statistics
         if( !paused )
             sprintf(status, "%.1f\n%d (%d)\n%.2f\n%.0f          \n%.2f\n%.2f (%2.0f it)\n%d\n%d\n%d",
                     mj_data_plot.time, mj_data_plot.nefc, mj_data_plot.ncon,
                     duration, 1.0/(glfwGetTime()-lastrendertm),
                     mj_data_plot.energy[0]+ mj_data_plot.energy[1],
                     mju_log10(mju_max(mjMINVAL, mj_data_plot.solverstat[1])),
                     mj_data_plot.solverstat[0],
                     cam.camid, vopt.frame, vopt.label );
        lastrendertm = glfwGetTime();
        
        // create geoms and lights
        mjv_makeGeoms(m, &mj_data_plot, &objects, &vopt, mjCAT_ALL, selbody,
                      (perturb & mjPERT_TRANSLATE) ? refpos : 0,
                      (perturb & mjPERT_ROTATE) ? refquat : 0, selpos);
        mjv_makeLights(m, &mj_data_plot, &objects);
        
        // update camera
        mjv_setCamera(m, &mj_data_plot, &cam);
        mjv_updateCameraPose(&cam, (mjtNum)rect.width/(mjtNum)rect.height);
        
        
        // render rgb
        mjr_render(0, rect, &objects, &ropt, &cam.pose, &con);
        
        
        // show overlays
        // if( showhelp==1 )
        //     mjr_overlay(rect, mjGRID_TOPLEFT, 0, "Help  ", "F1  ", &con);
        // else if( showhelp==2 )
        //     mjr_overlay(rect, mjGRID_TOPLEFT, 0, help_title, help_content, &con);
        
        // if( showinfo )
        // {
        //     if( paused )
        //         mjr_overlay(rect, mjGRID_BOTTOMLEFT, 0, "PAUSED", 0, &con);
        //     else
        //         mjr_overlay(rect, mjGRID_BOTTOMLEFT, 0,
        //             "Time\nSize\nCPU\nFPS\nEngy\nStat\nCam\nFrame\nLabel", status, &con);
        // }
        
        // if( showoption )
        // {
        //     int i;
        //     char buf[100];
        
        //     // fill titles on first pass
        //     if( !opt_title[0] )
        //     {
        //         for( i=0; i<mjNRNDFLAG; i++)
        //         {
        //             makeoptionstring(mjRNDSTRING[i][0], mjRNDSTRING[i][2][0], buf);
        //             strcat(opt_title, buf);
        //             strcat(opt_title, "\n");
        //         }
        //         for( i=0; i<mjNVISFLAG; i++)
        //         {
        //             makeoptionstring(mjVISSTRING[i][0], mjVISSTRING[i][2][0], buf);
        //             strcat(opt_title, buf);
        //             if( i<mjNVISFLAG-1 )
        //                 strcat(opt_title, "\n");
        //         }
        //     }
        
        //     // fill content
        //     opt_content[0] = 0;
        //     for( i=0; i<mjNRNDFLAG; i++)
        //     {
        //         strcat(opt_content, ropt.flags[i] ? " + " : "   ");
        //         strcat(opt_content, "\n");
        //     }
        //     for( i=0; i<mjNVISFLAG; i++)
        //     {
        //         strcat(opt_content, vopt.flags[i] ? " + " : "   ");
        //         if( i<mjNVISFLAG-1 )
        //             strcat(opt_content, "\n");
        //     }
        
        //     // show
        //     mjr_overlay(rect, mjGRID_TOPRIGHT, 0, opt_title, opt_content, &con);
        // }
        
        gui_mutex.unlock();
    }
    
    
    
double PyMJCWorld2::min_act(int act_index) {
    return m->actuator_user[m->nuser_actuator*act_index+3];
}
    
double PyMJCWorld2::max_act(int act_index) {
    return m->actuator_user[m->nuser_actuator*act_index+4];
}
    
int StateSize(mjModel* m) {
    return m->nq + m->nv + m->ntendon + m->ntendon;
}

void GetState(mjtNum* ptr, const mjModel* m, const mjData* d) {
    mju_copy(ptr, d->qpos, m->nq);
    ptr += m->nq;
    mju_copy(ptr, d->qvel, m->nv);
    ptr += m->nv;
    mju_copy(ptr, d->ten_length, m->ntendon);
    ptr += m->ntendon;
    mju_copy(ptr, d->ten_velocity, m->ntendon);
    
}
void SetState(const mjtNum* ptr, const mjModel* m, mjData* d) {
    mju_copy(d->qpos, ptr, m->nq);
    ptr += m->nq;
    mju_copy(d->qvel, ptr, m->nv);
    ptr += m->nv;
    mju_copy(d->ten_length, ptr, m->ntendon);
    ptr += m->ntendon;
    mju_copy(d->ten_velocity, ptr, m->ntendon);
    // include activations
}
    
inline void SetCtrl(const mjtNum* ptr, const mjModel* m, mjData* d) {
    mju_copy(d->ctrl, ptr, m->nu);
}

#define MJTNUM_DTYPE bn::dtype::get_builtin<mjtNum>()

bp::object PyMJCWorld2::Step(const bn::ndarray& x, const bn::ndarray& u) {
    //FAIL_IF_FALSE(x.get_dtype() == MJTNUM_DTYPE && x.get_nd() == 1 && x.get_flags() & bn::ndarray::C_CONTIGUOUS && x.shape(0) == m_model->nq+m_model->nv);
    //FAIL_IF_FALSE(u.get_dtype() == MJTNUM_DTYPE && u.get_nd() == 1 && u.get_flags() & bn::ndarray::C_CONTIGUOUS && u.shape(0) == m_model->nu);

    SetState(reinterpret_cast<const mjtNum*>(x.get_data()), m_model, m_data);

    mj_step1(m_model,m_data);
    SetCtrl(reinterpret_cast<const mjtNum*>(u.get_data()), m_model, m_data);
    mj_step2(m_model,m_data);

    long xdims[1] = {StateSize(m_model)};
    long site_dims[2] = {m_model->nsite, 3};
    bn::ndarray xout = bn::empty(1, xdims, bn::dtype::get_builtin<mjtNum>());
    bn::ndarray site_out = bn::empty(2, site_dims, bn::dtype::get_builtin<mjtNum>());

    GetState((mjtNum*)xout.get_data(), m_model, m_data);
    mju_copy((mjtNum*)site_out.get_data(), m_data->site_xpos, 3*m_model->nsite);

	return bp::make_tuple(xout, site_out);
}
    


void GetCOM(const mjModel* m, const mjData* d, mjtNum* com) {
    // see mj_com in engine_core.c
    mjtNum tot=0;
    com[0] = com[1] = com[2] = 0;
    for(int i=1; i<m->nbody; i++ ) {
        com[0] += d->xipos[3*i+0]*m->body_mass[i];
        com[1] += d->xipos[3*i+1]*m->body_mass[i];
        com[2] += d->xipos[3*i+2]*m->body_mass[i];
        tot += m->body_mass[i];
    }
    // compute com
    com[0] /= tot;
    com[1] /= tot;
    com[2] /= tot;
}

bn::ndarray PyMJCWorld2::GetCOMMulti(const bn::ndarray& x) {
    int state_size = StateSize(m_model);
    FAIL_IF_FALSE(x.get_dtype() == MJTNUM_DTYPE && x.get_nd() == 2 && x.get_flags() & bn::ndarray::C_CONTIGUOUS && x.shape(1) == state_size);
    int N = x.shape(0);
    long outdims[2] = {N,3};
    bn::ndarray out = bn::empty(2, outdims, bn::dtype::get_builtin<mjtNum>());
    mjtNum* ptr = (mjtNum*)out.get_data();
    for (int n=0; n < N; ++n) {
        SetState(reinterpret_cast<const mjtNum*>(x.get_data()), m_model, m_data);
        mj_kinematics(m_model, m_data);
        GetCOM(m_model, m_data, ptr);
        ptr += 3;
    }
    return out;
}

bn::ndarray PyMJCWorld2::GetJacSite(int site) {
    bn::ndarray out = bn::zeros(bp::make_tuple(3,m_model->nv), bn::dtype::get_builtin<mjtNum>());
    mjtNum* ptr = (mjtNum*)out.get_data();
    mj_jacSite(m_model, m_data, ptr, 0, site);
    return out;
}

void PyMJCWorld2::Kinematics() {
    mj_kinematics(m_model, m_data);
    mj_comPos(m_model, m_data);
    mj_tendon(m_model, m_data);
    mj_transmission(m_model, m_data);
}

void PyMJCWorld2::Plot(const bn::ndarray& x) {
    //FAIL_IF_FALSE(x.get_dtype() == MJTNUM_DTYPE && x.get_nd() == 1 && x.get_flags() & bn::ndarray::C_CONTIGUOUS);
    SetState(reinterpret_cast<const mjtNum*>(x.get_data()),m_model,&mj_data_plot);
    mj_forward(m_model, &mj_data_plot);
    //render(window);
}
    


void PyMJCWorld2::InitViewer(int width, int height, float cx, float cy, float cz, float px, float py, float pz) {
    int tempvar = 1;
}


int _ndarraysize(const bn::ndarray& arr) {
  int prod = 1;
  for (int i=0; i < arr.get_nd(); ++i) {
    prod *= arr.shape(i);
  }
  return prod;
}
template<typename T>
void _copyscalardata(const bp::object& from, T& to) {
  to = bp::extract<T>(from);
}
template <typename T>
void _copyarraydata(const bn::ndarray& from, T* to) {
  FAIL_IF_FALSE(from.get_dtype() == bn::dtype::get_builtin<T>() && from.get_flags() & bn::ndarray::C_CONTIGUOUS);
  memcpy(to, from.get_data(), _ndarraysize(from)*sizeof(T));
}
template<typename T>
void _csdihk(bp::dict d, const char* key, T& to) {
  // copy scalar data if has_key
  if (d.has_key(key)) _copyscalardata(d[key], to);
}
template<typename T>
void _cadihk(bp::dict d, const char* key, T* to) {
  // copy array data if has_key
  if (d.has_key(key)) {
    bn::ndarray arr = bp::extract<bn::ndarray>(d[key]);
    _copyarraydata<T>(arr, to);
  }
}

bp::dict PyMJCWorld2::GetModel() {
    bp::dict out;
    #include "mjcpy2_getmodel_autogen.i"
    return out;
}
void PyMJCWorld2::SetModel(bp::dict d) {
    #include "mjcpy2_setmodel_autogen.i"
}
bp::dict PyMJCWorld2::GetData() {
    bp::dict out;
    #include "mjcpy2_getdata_autogen.i"
    
    return out;
}
void PyMJCWorld2::SetData(bp::dict d) {
    #include "mjcpy2_setdata_autogen.i"
}


}



BOOST_PYTHON_MODULE(mjcpy) {
    bn::initialize();

    bp::class_<PyMJCWorld2,boost::noncopyable>("MJCWorld","docstring here", bp::init<const std::string&>())

        .def("step",&PyMJCWorld2::Step)
        .def("get_model",&PyMJCWorld2::GetModel)
        .def("set_model",&PyMJCWorld2::SetModel)
        .def("get_data",&PyMJCWorld2::GetData)
        .def("set_data",&PyMJCWorld2::SetData)
        .def("plot",&PyMJCWorld2::Plot)
        .def("init_viewer",&PyMJCWorld2::InitViewer)
        .def("get_COM_multi",&PyMJCWorld2::GetCOMMulti)
        .def("get_jac_site",&PyMJCWorld2::GetJacSite)
        .def("kinematics",&PyMJCWorld2::Kinematics)
        .def("set_num_steps",&PyMJCWorld2::SetNumSteps)
        .def("min_act",&PyMJCWorld2::min_act)
        .def("max_act",&PyMJCWorld2::max_act)
        ;


    bp::object main = bp::import("__main__");
    main_namespace = main.attr("__dict__");    
    bp::exec(
        "import numpy as np\n"
        "contact_dtype = np.dtype([('dim','i'), ('geom1','i'), ('geom2','i'),('flc_address','i'),('compliance','f8'),('timeconst','f8'),('dist','f8'),('mindist','f8'),('pos','f8',3),('frame','f8',9),('friction','f8',5)])\n"
        , main_namespace
    );


}
