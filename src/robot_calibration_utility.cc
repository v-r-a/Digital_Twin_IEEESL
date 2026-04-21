/* Drive the robot based on inputs from MuJoCo. Process the feedback to see
   implementation differences. This is a multi-thread program. One thread for
   simulation computations. One thread for simulation rendering. One thread for
   robot communication.
*/

// Following terminal commands work only for FT232R (USB to RS485 converter)
// based USB to serial bridge. Set the USB0 port eventchar to 129+256=385
// (10000001) to flush immediately on packet receipt. sudo -i echo 385 >
// /sys/bus/usb-serial/devices/ttyUSB0/event_char

// Set the USB port latency_timer to 10ms.
// echo 10 | sudo tee /sys/bus/usb-serial/devices/ttyUSB0/latency_timer

// sudo -i
// echo 385 > /sys/bus/usb-serial/devices/ttyUSB1/event_char
// Set the USB1 port latency_timer to 10ms.
// echo 10 | sudo tee /sys/bus/usb-serial/devices/ttyUSB1/latency_timer

// g++ cyber_physical_system.cc rs485_bus_functions.cc my_mjc_func.cc -O2
// -lpthread -lmujoco -lglfw  -o ../bin/cyber_physical_system

#include <GLFW/glfw3.h>
#include <stdint.h>
#include <sys/resource.h>
#include <vector>
#include <algorithm>
#include <deque>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <atomic>

#include "../include/mujoco/mujoco.h"
#include "SerialTransfer.h"
#include "my_mjc_func_declarations.h"
#include "rs485_bus_functions.h"
#include "exp_meas_proc_func.h"

// Function declarations

void communication_initialisation(int fsr_fresh[], int vref_fresh[]);
void communication_initialisation_robot(int fsr_today[], int vref_today[]);
void filter_robotData(std::deque<RobotDataToRead> &deque_readIn, RobotDataToRead *readIn);
void robot_reconstruction(int num, mjModel *mm, mjData *dSim, mjData *dExp, RobotDataToRead *read_in);

// Thread functions
void simulation(mjtNum *StateSimForRendering, mjtNum *StateSimForComm);
void communication_with_robot(mjtNum *StateSimSendToRobot, mjtNum *StateExpFromRobot);

void keyboard_cps(GLFWwindow *window, int key, int scancode, int act, int mods);
void mouse_button(GLFWwindow *window, int button, int act, int mods);
void mouse_move(GLFWwindow *window, double xpos, double ypos);
void scroll(GLFWwindow *window, double xoffset, double yoffset);

void init_qComp_figures(mjvFigure *myfig);
void update_qComp_figures(double q_des[], double q_sim[], double q_exp[]);
void render_qComp_figures(mjrRect vp);

void init_qErr_figures(mjvFigure *myfig);
void update_qErr_figures(double qErr_sim[], double qErr_exp[]);
void render_qErr_figures(mjrRect vp);

// void init_qDotErr_figures();
// void update_qDotErr_figures(double qDot_des[]);
// void render_qDotErr_figures(mjrRect vp);

// void init_fsr_figures();
// void update_fsr_figures();
// void render_fsr_figures(mjrRect vp);

// void init_torso_figures();
// void update_torso_figures();
// void render_torso_figures(mjrRect vp);

void init_exp_COP_plot(mjvFigure *myfig);
void update_exp_COP_plot(mjvFigure *myfig, mjModel *mm, mjData *dd, ContactDataAnalysis *cda, RobotDataToRead *readIn);
void render_exp_COP_plot(mjrRect vp);

void init_sim_COP_plot(mjvFigure *myfig);
void update_sim_COP_plot(mjvFigure *myfig, mjModel *mm, mjData *dd, ContactDataAnalysis *cda);
void render_sim_COP_plot(mjrRect vp_full);

void controller_pd(const mjModel *mm, mjData *dd, const double q_ctrl[], const double qdot_ctrl[],
                   const double KpGains[], const double KdGains[], const double sat_torques[],
                   const int qstartID, const int q_len);
void controller_wrapper();
void pose_updater(const mjModel *mm, mjData *dd);

// Colour definitions
float myRED[4] = {1, 0, 0, 0.8};
float myGREEN[4] = {0, 1, 0, 0.8};
float myBLUE[4] = {0, 0, 1, 0.8};
float myMAGENTA[4] = {1, 0, 1, 0.8};
float myCYAN[4] = {0, 1, 1, 0.8};
float myYELLOW[4] = {1, 1, 0, 0.8};

// Variables with dependencies outside the file
int vRef_today[9];       // 10-bit ADC Vref readings for both the feet and the force_stick.
int fsr_noLoad_today[9]; // 10-bit ADC preloaded sensor readings for both the feet and the force_stick.

// MuJoCo data structures
mjModel *m = NULL;             // MuJoCo model
mjData *d_sim = NULL;          // MuJoCo data corresponding to the simulation
ContactDataAnalysis mycda_sim; // Contact data analysis structure
mjData *d_exp = NULL;          // MuJoCo data corresponding to the experiment
ContactDataAnalysis mycda_exp; // Contact data analysis structure
mjvCamera cam;                 // abstract camera
mjvOption vOpt;                // visualization options (common for both the scenes)
mjvScene scn_sim;              // abstract scene corresponding to the simulation
mjvScene scn_exp;              // abstract scene corresponding to the experiment
mjrContext con;                // custom GPU context

// Synchronisation variables: sim and keyboard callbacks in main thread
std::timed_mutex mtx_d_sim;

// Synchronisation variables: sim and rendering
std::mutex mtx_state_sim_for_rendering;
std::condition_variable cv_rendering;
std::atomic_flag rendering_over = ATOMIC_FLAG_INIT;
std::atomic_uint time_rendering;
// Synchronisation variables: sim and communication
std::mutex mtx_state_sim_for_comm;
std::condition_variable cv_comm;
std::atomic_flag comm_over = ATOMIC_FLAG_INIT;
std::atomic_uint time_comm;

// Synchronisation variables: comm and rendering
std::timed_mutex mtx_state_exp;

mjvFigure fig_qComp[12];     // q comparison plots in all actuated DOFs
mjvFigure fig_qErr[12];      // q error plots in all actuated DOFs
mjvFigure fig_qDotErr[12];   // qdot error plots in all actuated DOFs
mjvFigure fig_fsr[8];        // FSR plots in all FSRs
mjvFigure fig_torso[3];      // Torso plots
mjvFigure fig_gcz_trace_exp; // pseudo-COP-ZMP plot trace exp
mjvFigure fig_gcz_trace_sim; // pseudo-COP-ZMP plot trace sim
enum
{
    FIG_none = 0,
    FIG_qComp_pg1,   // Joint angle comparison
    FIG_qComp_pg2,   // Joint angle comparison
    FIG_qErr_pg1,    // Joint angle error
    FIG_qErr_pg2,    // JOint angle error
    FIG_qDotErr_pg1, // Joint vel error
    FIG_qDotErr_pg2, // Joint vel error
    FIG_torso,       // Torso orientation error
    FIG_gcz_trace,   // pCOP-ZMP trace for simulation & experiment

    FIG_Npages
};
int fig_num = 0; // Figure page number

int filter_flag = 0;

int recons_num = 0;    // Reconstruction method number
int RECONS_Nmodes = 3; // Number of reconstruction methods

// mouse interaction
bool button_left = false;
bool button_middle = false;
bool button_right = false;
double lastx = 0;
double lasty = 0;

// Simulation related
bool start_stop_comm = 1;
bool ppause = false;
double example_motion_start_time = 1000.0;
int choosecontroller = 0;      // Choosing controller
int pushbody_id = TORSO_BODY;  // (Torso link) Body to apply push

// Only simulation thread reads and writes these variables.
double qjans_hold[12];  // qjans_hold is read out from file using load_qpos
double dqjans_hold[12]; // Set zero at start
double qjans[12];       // Used by IK function
double dqjans[12];      // Used by IK function

// For IK, CoM initialized through load_sitepose() and load_com_pos function
double txyz[3], tkphi[4], lfxyz[3], lfkphi[4], rfxyz[3], rfkphi[4], com_pos[3];

// PD Gains controlling the 12 DOFs of bioloid.
const double Kp = 7;
const double KpHip = 7;
const double Kd = 0.05;
const double Kp_hold[12] = {Kp, Kp, Kp * 1.5, Kp * 1.5, Kp, Kp, Kp, Kp, Kp * 1.5, Kp * 1.5, Kp, Kp};
const double Kd_hold[12] = {Kd, Kd, Kd, Kd, Kd / 5, Kd / 5, Kd, Kd, Kd, Kd, Kd / 5, Kd / 5};

// Saturation torques
const double Tsat = 1.3; // 1.3Nm stall torque, PITCH: 0.292 or 0.146 ROLL: 0.167 or 0.083 at ankle SSP/DSP
const double Tm = 5;
const double T_hold[12] = {Tsat, Tsat, Tsat, Tsat, Tsat, Tsat, Tsat, Tsat, Tsat, Tsat, Tsat, Tsat};
const double Tmax_hold[12] = {5, 5, 5, 5, Tsat, Tsat, 5, 5, 5, 5, Tsat, Tsat};
const double Tmax_1[12] = {Tm, Tm, Tm, Tm, Tm, Tm, Tm, Tm, Tm, Tm, Tm, Tm};

// IK utility
double incr = 0.001;    // increment for site x y z movements in the IK utility
double incr_ang = 0.05; // increment for  rotations in the IK utility

// Frame selector
bool tf = true;  // IK torso frame
bool lf = false; // IK left foot frame
bool rf = false; // IK right foot frame

// Functions scooped out to make the main() function look clean.
void init_qComp_figures(mjvFigure myfig[])
{
    // Plots of joint angles.
    char fig_legend1[100] = "des";
    char fig_legend2[100] = "sim";
    char fig_legend3[100] = "exp";
    // Order: {LHY_LL,LHY_UL,...,RAR_LL,RAR_UL}
    double fig_yrange[2] = {-0.1, 0.1};
    char fig_titles[12][6];
    // Titles: page 1
    mju_strncpy(fig_titles[0], "q LHY", 6);
    mju_strncpy(fig_titles[1], "q LHR", 6);
    mju_strncpy(fig_titles[2], "q LHP", 6);
    mju_strncpy(fig_titles[3], "q LKP", 6);
    mju_strncpy(fig_titles[4], "q LAP", 6);
    mju_strncpy(fig_titles[5], "q LAR", 6);
    // Titles: page 2
    mju_strncpy(fig_titles[6], "q RHY", 6);
    mju_strncpy(fig_titles[7], "q RHR", 6);
    mju_strncpy(fig_titles[8], "q RHP", 6);
    mju_strncpy(fig_titles[9], "q RKP", 6);
    mju_strncpy(fig_titles[10], "q RAP", 6);
    mju_strncpy(fig_titles[11], "q RAR", 6);

    // Initialise the figure structures.
    for (int i = 0; i < 12; i++)
        plot_d3t_init(&myfig[i], fig_titles[i], fig_legend1, fig_legend2, fig_legend3, 16, fig_yrange);
}
void update_qComp_figures(double q_des[], double q_sim[], double q_exp[])
{
    // Plots of joint angles.

    // Update the figure structures.
    double fig_yrange[2] = {-0.05, 0.05}, ypad = 0.05;
    ;
    // Upate page 1 or page 2 based on fig_num.
    switch (fig_num)
    {
    case FIG_qComp_pg1:
        // Update page 1 figures: 0-1-2 and 6-7-8
        for (int i = 0; i < 3; i++)
        {
            fig_yrange[0] = 0;
            fig_yrange[1] = 0;
            plot_d3t_update(&fig_qComp[i], 16, fig_yrange, q_des[i], q_sim[i], q_exp[i]);
        }
        for (int i = 6; i < 9; i++)
        {
            fig_yrange[0] = 0;
            fig_yrange[1] = 0;
            plot_d3t_update(&fig_qComp[i], 16, fig_yrange, q_des[i], q_sim[i], q_exp[i]);
        }
        break;
    case FIG_qComp_pg2:
        // Update page 2 figures: 3-4-5 and 9-10-11
        for (int i = 3; i < 6; i++)
        {
            fig_yrange[0] = 0;
            fig_yrange[1] = 0;
            plot_d3t_update(&fig_qComp[i], 16, fig_yrange, q_des[i], q_sim[i], q_exp[i]);
        }
        for (int i = 9; i < 12; i++)
        {
            fig_yrange[0] = 0;
            fig_yrange[1] = 0;
            plot_d3t_update(&fig_qComp[i], 16, fig_yrange, q_des[i], q_sim[i], q_exp[i]);
        }
        break;

    default:
        // Update none
        break;
    }
}
void render_qComp_figures(mjrRect vp_full)
{
    // Define window location for each plot.
    // Left leg plot in one column followed by right leg plots in the next column.
    // Only 6 figures are plotted on one page.
    mjrRect vp_qComp[6];

    // Left side error plots: 0-1-2 OR 3-4-5 (bottom to up)
    for (int i = 0; i < 3; i++)
    {
        // Height and width of each error plot.
        vp_qComp[i].width = vp_full.width / 4;
        vp_qComp[i].height = vp_full.height / 3;

        // Location of the bottom left corner of each plot.
        vp_qComp[i].left = vp_full.left + vp_full.width / 2;
        vp_qComp[i].bottom = vp_full.bottom + (i * vp_full.height / 3);
    }

    // Right side error plots: 6-7-8 OR 9-10-11 (bottom to up)
    for (int i = 3; i < 6; i++)
    {
        // Height and width of each error plot.
        vp_qComp[i].width = vp_full.width / 4;
        vp_qComp[i].height = vp_full.height / 3;

        // Location of the bottom left corner of each plot.
        vp_qComp[i].left = vp_full.left + vp_full.width / 2 + vp_full.width / 4;
        vp_qComp[i].bottom = vp_full.bottom + ((i - 3) * vp_full.height / 3);
    }

    // Render page 1 or page 2 based on fig_num.
    switch (fig_num)
    {
    case FIG_qComp_pg1:
        // Render page 1 figures
        for (int i = 0; i < 3; i++)
            mjr_figure(vp_qComp[i], &fig_qComp[i], &con);
        for (int i = 6; i < 9; i++)
            mjr_figure(vp_qComp[i - 3], &fig_qComp[i], &con);
        break;
    case FIG_qComp_pg2:
        // Render page 2 figures
        for (int i = 3; i < 6; i++)
            mjr_figure(vp_qComp[i - 3], &fig_qComp[i], &con);
        for (int i = 9; i < 12; i++)
            mjr_figure(vp_qComp[i - 6], &fig_qComp[i], &con);
        break;
    default:
        // Render none
        break;
    }
}
void init_qErr_figures(mjvFigure myfig[])
{
    // Plots of errors in joint angles (in deg).
    char fig_legend1[100] = "d-s";
    char fig_legend2[100] = "s-e";
    // Order: {LHY_LL,LHY_UL,...,RAR_LL,RAR_UL}
    double fig_yrange[2] = {-3, 3};
    char fig_titles[12][9];
    // Titles: page 1
    mju_strncpy(fig_titles[0], "qErr LHY", 9);
    mju_strncpy(fig_titles[1], "qErr LHR", 9);
    mju_strncpy(fig_titles[2], "qErr LHP", 9);
    mju_strncpy(fig_titles[3], "qErr LKP", 9);
    mju_strncpy(fig_titles[4], "qErr LAP", 9);
    mju_strncpy(fig_titles[5], "qErr LAR", 9);
    // Titles: page 2
    mju_strncpy(fig_titles[6], "qErr RHY", 9);
    mju_strncpy(fig_titles[7], "qErr RHR", 9);
    mju_strncpy(fig_titles[8], "qErr RHP", 9);
    mju_strncpy(fig_titles[9], "qErr RKP", 9);
    mju_strncpy(fig_titles[10], "qErr RAP", 9);
    mju_strncpy(fig_titles[11], "qErr RAR", 9);

    // Initialise the figure structures.
    for (int i = 0; i < 12; i++)
        plot_d2t_init(&myfig[i], fig_titles[i], fig_legend1, fig_legend2, 16, fig_yrange);
}
void update_qErr_figures(double qErr_sim[], double qErr_exp[])
{
    // Plots of errors in joint angles (in deg).

    // Update the figure structures.
    double fig_yrange[2] = {-3, 3};
    // Upate page 1 or page 2 based on fig_num.
    switch (fig_num)
    {
    case FIG_qErr_pg1:
        // Update page 1 figures: 0-1-2 and 6-7-8
        for (int i = 0; i < 3; i++)
            plot_d2t_update(&fig_qErr[i], 16, fig_yrange, qErr_sim[i], qErr_exp[i]);
        for (int i = 6; i < 9; i++)
            plot_d2t_update(&fig_qErr[i], 16, fig_yrange, qErr_sim[i], qErr_exp[i]);
        break;
    case FIG_qErr_pg2:
        // Update page 2 figures: 3-4-5 and 9-10-11
        for (int i = 3; i < 6; i++)
            plot_d2t_update(&fig_qErr[i], 16, fig_yrange, qErr_sim[i], qErr_exp[i]);
        for (int i = 9; i < 12; i++)
            plot_d2t_update(&fig_qErr[i], 16, fig_yrange, qErr_sim[i], qErr_exp[i]);
        break;

    default:
        // Update none
        break;
    }
}
void render_qErr_figures(mjrRect vp_full)
{
    // Define window location for each plot.
    // Left leg plot in one column followed by right leg plots in the next column.
    // Only 6 figures are plotted on one page.
    mjrRect vp_qErr[6];

    // Left side error plots: 0-1-2 OR 3-4-5 (bottom to up)
    for (int i = 0; i < 3; i++)
    {
        // Height and width of each error plot.
        vp_qErr[i].width = vp_full.width / 4;
        vp_qErr[i].height = vp_full.height / 3;

        // Location of the bottom left corner of each plot.
        vp_qErr[i].left = vp_full.left + vp_full.width / 2;
        vp_qErr[i].bottom = vp_full.bottom + (i * vp_full.height / 3);
    }

    // Right side error plots: 6-7-8 OR 9-10-11 (bottom to up)
    for (int i = 3; i < 6; i++)
    {
        // Height and width of each error plot.
        vp_qErr[i].width = vp_full.width / 4;
        vp_qErr[i].height = vp_full.height / 3;

        // Location of the bottom left corner of each plot.
        vp_qErr[i].left = vp_full.left + vp_full.width / 2 + vp_full.width / 4;
        vp_qErr[i].bottom = vp_full.bottom + ((i - 3) * vp_full.height / 3);
    }

    // Render page 1 or page 2 based on fig_num.
    switch (fig_num)
    {
    case FIG_qErr_pg1:
        // Render page 1 figures
        for (int i = 0; i < 3; i++)
            mjr_figure(vp_qErr[i], &fig_qErr[i], &con);
        for (int i = 6; i < 9; i++)
            mjr_figure(vp_qErr[i - 3], &fig_qErr[i], &con);
        break;
    case FIG_qErr_pg2:
        // Render page 2 figures
        for (int i = 3; i < 6; i++)
            mjr_figure(vp_qErr[i - 3], &fig_qErr[i], &con);
        for (int i = 9; i < 12; i++)
            mjr_figure(vp_qErr[i - 6], &fig_qErr[i], &con);
        break;
    default:
        // Render none
        break;
    }
}
void init_sim_COP_plot(mjvFigure *myfig)
{
    // This function (initiates) plots the projection of the generalised COP-ZOP
    // and the foot support polygon.

    // set figure to default
    mjv_defaultFigure(myfig);

    // title
    char mytitle[1000] = "Sim COP-ZMP projection trace";
    mju_strncpy(myfig->title, mytitle, 1000);

    // x-y tick number formats
    mju_strncpy(myfig->xformat, "%.2f", 20);
    mju_strncpy(myfig->yformat, "%.2f", 20);

    // grid sizes
    myfig->gridsize[0] = 5;
    myfig->gridsize[1] = 5;

    // number of points on a line: myfig->linepnt[0] = 960 /* mjMAXLINEPNT */;
    // 8 lines for two robot feet
    for (int i = 0; i < 8; i++)
    {
        myfig->linepnt[i] = 500;
    }
    // the COP-ZMP projection trace
    myfig->linepnt[8] = 2 /* mjMAXLINEPNT */;

    // Line RGB
    float footlineRGB[3] = {0, 1, 0};
    for (int i = 0; i < 8; i++)
    {
        memcpy(myfig->linergb[i], footlineRGB, sizeof(float) * 3);
    }

    // Set plot X-Y range
    myfig->range[0][0] = -0.060;
    myfig->range[0][1] = 0.060;
    myfig->range[1][0] = -0.060;
    myfig->range[1][1] = 0.060;

    // Foot contact points initial values:
    float xa = 0.055;
    float xb = 0.010;
    float ya = 0.035;
    float FootVertices[8][2] = {{-xa, -ya}, {-xa, ya}, {-xb, ya}, {-xb, -ya}, {xa, -ya}, {xa, ya}, {xb, ya}, {xb, -ya}};

    // Draw feet
    my_linspace_2d(FootVertices[0], FootVertices[1], 500, myfig->linedata[0]);
    my_linspace_2d(FootVertices[1], FootVertices[2], 500, myfig->linedata[1]);
    my_linspace_2d(FootVertices[2], FootVertices[3], 500, myfig->linedata[2]);
    my_linspace_2d(FootVertices[3], FootVertices[0], 500, myfig->linedata[3]);

    my_linspace_2d(FootVertices[4], FootVertices[5], 500, myfig->linedata[4]);
    my_linspace_2d(FootVertices[5], FootVertices[6], 500, myfig->linedata[5]);
    my_linspace_2d(FootVertices[6], FootVertices[7], 500, myfig->linedata[6]);
    my_linspace_2d(FootVertices[7], FootVertices[4], 500, myfig->linedata[7]);
}
void update_sim_COP_plot(mjvFigure *myfig, mjModel *mm, mjData *dd, ContactDataAnalysis *cda)
{
    // Some line joining bug when less than 8 contact points.
    // This function updates the projection of the generalised COP-ZOP and the
    // foot support polygon.

    // Update only if fig_num is 4
    if (fig_num != FIG_gcz_trace)
        return;

    // Set plot X-Y range
    myfig->range[0][0] = -0.060;
    myfig->range[0][1] = 0.060;
    myfig->range[1][0] = -0.060;
    myfig->range[1][1] = 0.060;

    // Foot contact points initial values:
    float xa = 0.055;
    float xb = 0.010;
    float ya = 0.035;
    float FootVertices[8][2] = {{-xa, -ya}, {-xa, ya}, {-xb, ya}, {-xb, -ya}, {xa, -ya}, {xa, ya}, {xb, ya}, {xb, -ya}};

    // Updated values
    int ngrp = cda->ngroups; // number of contact groups
    int cgrpId[dd->ncon];    // contact id where new group starts
    memcpy(cgrpId, cda->group_start_IDs, sizeof(int) * dd->ncon);
    int nc_ingrp[dd->ncon]; // number of contacts in the group
    memcpy(nc_ingrp, cda->ncon_in_group, sizeof(int) * dd->ncon);

    if (dd->ncon)
    {
        int k = 0;
        if (ngrp < 3)
        {
            for (int i = 0; i < dd->ncon; i++)
            {
                FootVertices[i][0] = dd->contact[i].pos[0]; // copy x coord
                FootVertices[i][1] = dd->contact[i].pos[1]; // copy y coord
            }

            // How many line segments?
            for (int i = 0; i < ngrp; i++)
            {
                // loop over contacts in that group
                for (int j = 0; j < (nc_ingrp[i] - 1); j++)
                {
                    k++;
                }
                // Line joining the last point in contact group to the first point in
                // the contact group
                k++;
            }
            for (int i = 0; i < k; i++)
            {
                myfig->linepnt[i] = 500;
            }

            // Draw feet
            // loop over groups
            k = 0;
            for (int i = 0; i < ngrp; i++)
            {
                // loop over contacts in that group
                for (int j = 0; j < (nc_ingrp[i] - 1); j++)
                {
                    my_linspace_2d(FootVertices[cgrpId[i] + j],
                                   FootVertices[cgrpId[i] + j + 1], 500,
                                   myfig->linedata[k]);
                    k++;
                }
                // Line joining the last point in contact group to the first point in
                // the contact group
                my_linspace_2d(FootVertices[cgrpId[i] + nc_ingrp[i] - 1],
                               FootVertices[cgrpId[i]], 500, myfig->linedata[k]);
                k++;
            }

            // Foot support polygon edge colour
            float footlineRGB[3] = {0, 1, 0};
            for (int i = 0; i < k; i++)
            {
                memcpy(myfig->linergb[i], footlineRGB, sizeof(float) * 3);
            }
        }
        else
        {
            // decide later. plot CHull of the support polygon
        }

        // COP-ZMP trace
        myfig->linepnt[k] = 500;

        // COP-ZMP colour
        float footlineRGB[3] = {1, 0, 0};
        memcpy(myfig->linergb[k], footlineRGB, sizeof(float) * 3);

        // Centroid of the projected foot support polygon
        float centroid[2] = {0.0, 0.0};
        for (int i = 0; i < dd->ncon; i++)
        {
            centroid[0] += FootVertices[i][0];
            centroid[1] += FootVertices[i][1];
        }
        centroid[0] /= dd->ncon;
        centroid[1] /= dd->ncon;
        myfig->linedata[k][0] = centroid[0];
        myfig->linedata[k][1] = centroid[1];

        // The trace
        for (int i = 500 - 1; i > 1; i--)
        {
            myfig->linedata[k][2 * i] = myfig->linedata[k][2 * (i - 1)];
            myfig->linedata[k][2 * i + 1] = myfig->linedata[k][2 * (i - 1) + 1];
        }

        // Plot the latest experiment COP-ZMP projection
        double gcz[3] = {0, 0, 0};
        mju_copy3(gcz, cda->gcop);

        myfig->linedata[k][2] = gcz[0];
        myfig->linedata[k][3] = gcz[1];
    }
}
void render_sim_COP_plot(mjrRect vp_full)
{
    mjrRect viewport_fig_gcz_trace = {vp_full.left + vp_full.width / 2, vp_full.bottom + vp_full.height / 2, vp_full.width / 2, vp_full.height / 2};

    switch (fig_num)
    {
    case FIG_gcz_trace:
        mjr_figure(viewport_fig_gcz_trace, &fig_gcz_trace_sim, &con);
        break;
    default:
        // None
        break;
    }
}
void init_exp_COP_plot(mjvFigure *myfig)
{
    // This function (initiates) plots the projection of the generalised COP-ZOP
    // and the foot support polygon.

    // set figure to default
    mjv_defaultFigure(myfig);

    // title
    char mytitle[1000] = "Exp COP-ZMP projection trace";
    mju_strncpy(myfig->title, mytitle, 1000);

    // x-y tick number formats
    mju_strncpy(myfig->xformat, "%.2f", 20);
    mju_strncpy(myfig->yformat, "%.2f", 20);

    // grid sizes
    myfig->gridsize[0] = 5;
    myfig->gridsize[1] = 5;

    // number of points on a line: myfig->linepnt[0] = 960 /* mjMAXLINEPNT */;
    // 8 lines for two robot feet
    for (int i = 0; i < 8; i++)
    {
        myfig->linepnt[i] = 500;
    }
    // the COP-ZMP projection trace
    myfig->linepnt[8] = 2 /* mjMAXLINEPNT */;

    // Line RGB
    float footlineRGB[3] = {0, 1, 0};
    for (int i = 0; i < 8; i++)
    {
        memcpy(myfig->linergb[i], footlineRGB, sizeof(float) * 3);
    }

    // Set plot X-Y range
    myfig->range[0][0] = -0.060;
    myfig->range[0][1] = 0.060;
    myfig->range[1][0] = -0.060;
    myfig->range[1][1] = 0.060;

    // Foot contact points initial values:
    float xa = 0.055;
    float xb = 0.010;
    float ya = 0.035;
    float FootVertices[8][2] = {{-xa, -ya}, {-xa, ya}, {-xb, ya}, {-xb, -ya}, {xa, -ya}, {xa, ya}, {xb, ya}, {xb, -ya}};

    // Draw feet
    my_linspace_2d(FootVertices[0], FootVertices[1], 500, myfig->linedata[0]);
    my_linspace_2d(FootVertices[1], FootVertices[2], 500, myfig->linedata[1]);
    my_linspace_2d(FootVertices[2], FootVertices[3], 500, myfig->linedata[2]);
    my_linspace_2d(FootVertices[3], FootVertices[0], 500, myfig->linedata[3]);

    my_linspace_2d(FootVertices[4], FootVertices[5], 500, myfig->linedata[4]);
    my_linspace_2d(FootVertices[5], FootVertices[6], 500, myfig->linedata[5]);
    my_linspace_2d(FootVertices[6], FootVertices[7], 500, myfig->linedata[6]);
    my_linspace_2d(FootVertices[7], FootVertices[4], 500, myfig->linedata[7]);
}
void update_exp_COP_plot(mjvFigure *myfig, mjModel *mm, mjData *dd, ContactDataAnalysis *cda, RobotDataToRead *readIn)
{
    // Some line joining bug when less than 8 contact points.
    // This function updates the projection of the generalised COP-ZOP and the
    // foot support polygon.

    // Update only if fig_num is 4
    if (fig_num != FIG_gcz_trace)
        return;

    // Set plot X-Y range
    myfig->range[0][0] = -0.060;
    myfig->range[0][1] = 0.060;
    myfig->range[1][0] = -0.060;
    myfig->range[1][1] = 0.060;

    // Foot contact points initial values:
    float xa = 0.055;
    float xb = 0.010;
    float ya = 0.035;
    float FootVertices[8][2] = {{-xa, -ya}, {-xa, ya}, {-xb, ya}, {-xb, -ya}, {xa, -ya}, {xa, ya}, {xb, ya}, {xb, -ya}};

    // Updated values
    int ngrp = cda->ngroups; // number of contact groups
    int cgrpId[dd->ncon];    // contact id where new group starts
    memcpy(cgrpId, cda->group_start_IDs, sizeof(int) * dd->ncon);
    int nc_ingrp[dd->ncon]; // number of contacts in the group
    memcpy(nc_ingrp, cda->ncon_in_group, sizeof(int) * dd->ncon);

    if (dd->ncon)
    {
        int k = 0;
        if (ngrp < 3)
        {
            for (int i = 0; i < dd->ncon; i++)
            {
                FootVertices[i][0] = dd->contact[i].pos[0]; // copy x coord
                FootVertices[i][1] = dd->contact[i].pos[1]; // copy y coord
            }

            // How many line segments?
            for (int i = 0; i < ngrp; i++)
            {
                // loop over contacts in that group
                for (int j = 0; j < (nc_ingrp[i] - 1); j++)
                {
                    k++;
                }
                // Line joining the last point in contact group to the first point in
                // the contact group
                k++;
            }
            for (int i = 0; i < k; i++)
            {
                myfig->linepnt[i] = 500;
            }

            // Draw feet
            // loop over groups
            k = 0;
            for (int i = 0; i < ngrp; i++)
            {
                // loop over contacts in that group
                for (int j = 0; j < (nc_ingrp[i] - 1); j++)
                {
                    my_linspace_2d(FootVertices[cgrpId[i] + j],
                                   FootVertices[cgrpId[i] + j + 1], 500,
                                   myfig->linedata[k]);
                    k++;
                }
                // Line joining the last point in contact group to the first point in
                // the contact group
                my_linspace_2d(FootVertices[cgrpId[i] + nc_ingrp[i] - 1],
                               FootVertices[cgrpId[i]], 500, myfig->linedata[k]);
                k++;
            }

            // Foot support polygon edge colour
            float footlineRGB[3] = {0, 1, 0};
            for (int i = 0; i < k; i++)
            {
                memcpy(myfig->linergb[i], footlineRGB, sizeof(float) * 3);
            }
        }
        else
        {
            // decide later. plot CHull of the support polygon
        }

        // COP-ZMP trace
        myfig->linepnt[k] = 500;

        // COP-ZMP colour
        float footlineRGB[3] = {1, 0, 0};
        memcpy(myfig->linergb[k], footlineRGB, sizeof(float) * 3);

        // Centroid of the projected foot support polygon
        float centroid[2] = {0.0, 0.0};
        for (int i = 0; i < dd->ncon; i++)
        {
            centroid[0] += FootVertices[i][0];
            centroid[1] += FootVertices[i][1];
        }
        centroid[0] /= dd->ncon;
        centroid[1] /= dd->ncon;
        myfig->linedata[k][0] = centroid[0];
        myfig->linedata[k][1] = centroid[1];

        // The trace
        for (int i = 500 - 1; i > 1; i--)
        {
            myfig->linedata[k][2 * i] = myfig->linedata[k][2 * (i - 1)];
            myfig->linedata[k][2 * i + 1] = myfig->linedata[k][2 * (i - 1) + 1];
        }

        // Plot the latest experiment COP-ZMP projection
        double gcz[3] = {0, 0, 0}, lf_gcz[3], rf_gcz[3];
        // exp_genCOP(mm, dd, readIn, lf_gcz, rf_gcz, gcz);
        exp_genCOP1(mm, dd, readIn, 0, 0, lf_gcz, rf_gcz, gcz);
        // printf("expgcz: %f %f %f\n", gcz[0], gcz[1], gcz[2]);
        myfig->linedata[k][2] = gcz[0];
        myfig->linedata[k][3] = gcz[1];
    }
}
void render_exp_COP_plot(mjrRect vp_full)
{
    mjrRect viewport_fig_gcz_trace = {vp_full.left + vp_full.width / 2, vp_full.bottom, vp_full.width / 2, vp_full.height / 2};

    switch (fig_num)
    {
    case FIG_gcz_trace:
        mjr_figure(viewport_fig_gcz_trace, &fig_gcz_trace_exp, &con);
        break;
    default:
        // None
        break;
    }
}

void my_visuals_initialisations()
{
    // initialize visualization data structures
    mjv_defaultCamera(&cam);
    mjv_defaultOption(&vOpt);

    mjv_defaultScene(&scn_sim);
    mjv_defaultScene(&scn_exp);
    mjr_defaultContext(&con);

    // create scene and context
    mjv_makeScene(m, &scn_sim, 500);
    mjv_makeScene(m, &scn_exp, 500);
    mjr_makeContext(m, &con, mjFONTSCALE_150);

    // Initialise the plots/figures.
    init_qComp_figures(fig_qComp);
    init_qErr_figures(fig_qErr);
    init_sim_COP_plot(&fig_gcz_trace_sim);
    init_exp_COP_plot(&fig_gcz_trace_exp);

    // Rendering preferences:
    scn_sim.flags[mjRND_SHADOW] = false;
    scn_sim.flags[mjRND_REFLECTION] = false;
    scn_exp.flags[mjRND_SHADOW] = false;
    scn_exp.flags[mjRND_REFLECTION] = false;
}

void my_glfw_initialisations(GLFWwindow *win)
{
    glfwMakeContextCurrent(win);
    glfwSwapInterval(1); // Rendering sync with monitor (v-sync), i.e., 60 fps

    // install GLFW mouse and keyboard callbacks
    glfwSetKeyCallback(win, keyboard_cps);
    glfwSetCursorPosCallback(win, mouse_move);
    glfwSetMouseButtonCallback(win, mouse_button);
    glfwSetScrollCallback(win, scroll);
}

void render_text(mjModel *mm, mjData *dd, mjData *dd_fdbk, RobotDataToRead *readIn_for_rendering1, mjrRect vp_full)
{
    char text[200];

    // Define the viewport dimensions and the initial position
    mjrRect vp_text;
    vp_text.left = vp_full.left + vp_full.width / 2;
    vp_text.bottom = vp_full.bottom + vp_full.height * 12 / 13;
    vp_text.width = vp_full.width / 2;
    vp_text.height = vp_full.height / 26;

    float lineSpacing = 1;
    // Background and text colors
    float bgColor[4] = {0.2f, 0.2f, 0.2f, 0.8f}; // Dark grey with some transparency
    float textColor[3] = {1.0f, 1.0f, 1.0f};     // White text

    /********************************************************************************************************************************************/

    // Format and render txyz
    std::sprintf(text, "txyz: (%.3f, %.3f, %.3f)", txyz[0], txyz[1], txyz[2]);
    mjr_label(vp_text, mjFONT_NORMAL, text, bgColor[0], bgColor[1], bgColor[2], bgColor[3], textColor[0], textColor[1], textColor[2], &con);

    // Move to the next line and adjust the viewport
    vp_text.bottom -= static_cast<int>(vp_text.height * lineSpacing);

    // Format and render tkphi
    std::sprintf(text, "tkphi: (%.3f, %.3f, %.3f, %.3f)", tkphi[0], tkphi[1], tkphi[2], tkphi[3]);
    mjr_label(vp_text, mjFONT_NORMAL, text, bgColor[0], bgColor[1], bgColor[2], bgColor[3], textColor[0], textColor[1], textColor[2], &con);

    // Move to the next line and adjust the viewport
    vp_text.bottom -= static_cast<int>(vp_text.height * lineSpacing);

    // Format and render lfxyz
    std::sprintf(text, "lfxyz: (%.3f, %.3f, %.3f)", lfxyz[0], lfxyz[1], lfxyz[2]);
    mjr_label(vp_text, mjFONT_NORMAL, text, bgColor[0], bgColor[1], bgColor[2], bgColor[3], textColor[0], textColor[1], textColor[2], &con);

    // Move to the next line and adjust the viewport
    vp_text.bottom -= static_cast<int>(vp_text.height * lineSpacing);

    // Format and render lfkphi
    std::sprintf(text, "lfkphi: (%.3f, %.3f, %.3f, %.3f)", lfkphi[0], lfkphi[1], lfkphi[2], lfkphi[3]);
    mjr_label(vp_text, mjFONT_NORMAL, text, bgColor[0], bgColor[1], bgColor[2], bgColor[3], textColor[0], textColor[1], textColor[2], &con);

    // Move to the next line and adjust the viewport
    vp_text.bottom -= static_cast<int>(vp_text.height * lineSpacing);

    // Format and render rfxyz
    std::sprintf(text, "rfxyz: (%.3f, %.3f, %.3f)", rfxyz[0], rfxyz[1], rfxyz[2]);
    mjr_label(vp_text, mjFONT_NORMAL, text, bgColor[0], bgColor[1], bgColor[2], bgColor[3], textColor[0], textColor[1], textColor[2], &con);

    // Move to the next line and adjust the viewport
    vp_text.bottom -= static_cast<int>(vp_text.height * lineSpacing);

    // Format and render rfkphi
    std::sprintf(text, "rfkphi: (%.3f, %.3f, %.3f, %.3f)", rfkphi[0], rfkphi[1], rfkphi[2], rfkphi[3]);
    mjr_label(vp_text, mjFONT_NORMAL, text, bgColor[0], bgColor[1], bgColor[2], bgColor[3], textColor[0], textColor[1], textColor[2], &con);

    /********************************************************************************************************************************************/

    // Move to the next line and adjust the viewport
    vp_text.bottom -= static_cast<int>(vp_text.height * lineSpacing);

    // Format and render floating qpos[0-2]
    std::sprintf(text, "Floating base XYZ: (%.3f, %.3f, %.3f)", dd->qpos[0], dd->qpos[1], dd->qpos[2]);
    mjr_label(vp_text, mjFONT_NORMAL, text, bgColor[0], bgColor[1], bgColor[2], bgColor[3], textColor[0], textColor[1], textColor[2], &con);

    // Move to the next line and adjust the viewport
    vp_text.bottom -= static_cast<int>(vp_text.height * lineSpacing);

    // Format and render floating qpos[3-6]
    std::sprintf(text, "Floating base Quat: (%.3f, %.3f, %.3f, %.3f)", dd->qpos[3], dd->qpos[4], dd->qpos[5], dd->qpos[6]);
    mjr_label(vp_text, mjFONT_NORMAL, text, bgColor[0], bgColor[1], bgColor[2], bgColor[3], textColor[0], textColor[1], textColor[2], &con);

    // Move to the next line and adjust the viewport
    vp_text.bottom -= static_cast<int>(vp_text.height * lineSpacing);

    // Format and render left foot joint angles qpos[7-12]
    std::sprintf(text, "Left foot joint angles: (%.3f, %.3f, %.3f, %.3f, %.3f, %.3f)", dd->qpos[7], dd->qpos[8], dd->qpos[9], dd->qpos[10], dd->qpos[11], dd->qpos[12]);
    mjr_label(vp_text, mjFONT_NORMAL, text, bgColor[0], bgColor[1], bgColor[2], bgColor[3], textColor[0], textColor[1], textColor[2], &con);

    // Move to the next line and adjust the viewport
    vp_text.bottom -= static_cast<int>(vp_text.height * lineSpacing);

    // Format and render right foot joint angles qpos[13-18]
    std::sprintf(text, "Right foot joint angles: (%.3f, %.3f, %.3f, %.3f, %.3f, %.3f)", dd->qpos[13], dd->qpos[14], dd->qpos[15], dd->qpos[16], dd->qpos[17], dd->qpos[18]);
    mjr_label(vp_text, mjFONT_NORMAL, text, bgColor[0], bgColor[1], bgColor[2], bgColor[3], textColor[0], textColor[1], textColor[2], &con);

    /********************************************************************************************************************************************/

    uint16_t qpos_10bit[12];
    map_angles_mjc_to_robot(dd->qpos + 7, qpos_10bit);
    uint16_t qpos_10bit_exp[12];
    map_angles_mjc_to_robot(dd_fdbk->qpos + 7, qpos_10bit_exp); // Double inversion 10-bit -> mjc -> 10-bit

    // Move to the next line and adjust the viewport
    vp_text.bottom -= static_cast<int>(vp_text.height * lineSpacing);

    // Format and render QPOS in 10-bit values for Left leg
    std::sprintf(text, "PC LL 10-bit: (%u, %u, %u, %u, %u, %u)", qpos_10bit[0], qpos_10bit[1], qpos_10bit[2], qpos_10bit[3], qpos_10bit[4], qpos_10bit[5]);
    mjr_label(vp_text, mjFONT_NORMAL, text, bgColor[0], bgColor[1], bgColor[2], bgColor[3], textColor[0], textColor[1], textColor[2], &con);

    // Move to the next line and adjust the viewport
    vp_text.bottom -= static_cast<int>(vp_text.height * lineSpacing);

    // Format and render the qpos from the robot for the left foot
    std::sprintf(text, "RE LL 10-bit: (%u, %u, %u, %u, %u, %u)", qpos_10bit_exp[0], qpos_10bit_exp[1], qpos_10bit_exp[2], qpos_10bit_exp[3], qpos_10bit_exp[4], qpos_10bit_exp[5]);
    mjr_label(vp_text, mjFONT_NORMAL, text, bgColor[0], bgColor[1], bgColor[2], bgColor[3], textColor[0], textColor[1], textColor[2], &con);

    // Move to the next line and adjust the viewport
    vp_text.bottom -= static_cast<int>(vp_text.height * lineSpacing);

    // Format and render QPOS in 10-bit values for Right leg
    std::sprintf(text, "PC RL 10-bit: (%u, %u, %u, %u, %u, %u)", qpos_10bit[6], qpos_10bit[7], qpos_10bit[8], qpos_10bit[9], qpos_10bit[10], qpos_10bit[11]);
    mjr_label(vp_text, mjFONT_NORMAL, text, bgColor[0], bgColor[1], bgColor[2], bgColor[3], textColor[0], textColor[1], textColor[2], &con);

    // Move to the next line and adjust the viewport
    vp_text.bottom -= static_cast<int>(vp_text.height * lineSpacing);

    // Format and render the qpos from the robot for the right foot
    std::sprintf(text, "RE RL 10-bit: (%u, %u, %u, %u, %u, %u)", qpos_10bit_exp[6], qpos_10bit_exp[7], qpos_10bit_exp[8], qpos_10bit_exp[9], qpos_10bit_exp[10], qpos_10bit_exp[11]);
    mjr_label(vp_text, mjFONT_NORMAL, text, bgColor[0], bgColor[1], bgColor[2], bgColor[3], textColor[0], textColor[1], textColor[2], &con);

    /********************************************************************************************************************************************/

    // Move to the next line and adjust the viewport
    vp_text.bottom -= static_cast<int>(vp_text.height * lineSpacing);

    // Format and render the torso orientation from IMU
    std::sprintf(text, "IMU Torso Quat: (%.3f, %.3f, %.3f, %.3f)", readIn_for_rendering1->torso_quat[0], readIn_for_rendering1->torso_quat[1], readIn_for_rendering1->torso_quat[2], readIn_for_rendering1->torso_quat[3]);
    mjr_label(vp_text, mjFONT_NORMAL, text, bgColor[0], bgColor[1], bgColor[2], bgColor[3], textColor[0], textColor[1], textColor[2], &con);

    // Move to the next line and adjust the viewport
    vp_text.bottom -= static_cast<int>(vp_text.height * lineSpacing);

    // Format and render the left foot orientation from IMU
    std::sprintf(text, "IMU LF Quat: (%.3f, %.3f, %.3f, %.3f)", readIn_for_rendering1->lf_quat[0], readIn_for_rendering1->lf_quat[1], readIn_for_rendering1->lf_quat[2], readIn_for_rendering1->lf_quat[3]);
    mjr_label(vp_text, mjFONT_NORMAL, text, bgColor[0], bgColor[1], bgColor[2], bgColor[3], textColor[0], textColor[1], textColor[2], &con);

    // Move to the next line and adjust the viewport
    vp_text.bottom -= static_cast<int>(vp_text.height * lineSpacing);

    // Format and render the right foot orientation from IMU
    std::sprintf(text, "IMU RF Quat: (%.3f, %.3f, %.3f, %.3f)", readIn_for_rendering1->rf_quat[0], readIn_for_rendering1->rf_quat[1], readIn_for_rendering1->rf_quat[2], readIn_for_rendering1->rf_quat[3]);
    mjr_label(vp_text, mjFONT_NORMAL, text, bgColor[0], bgColor[1], bgColor[2], bgColor[3], textColor[0], textColor[1], textColor[2], &con);

    /********************************************************************************************************************************************/

    // Move to the next line and adjust the viewport
    vp_text.bottom -= static_cast<int>(vp_text.height * lineSpacing);

    // Calculate (k,phi) from the quaternion
    double kphi[4];
    my_mju_quat2axisAngle(readIn_for_rendering1->torso_quat, kphi);

    // Format and render the torso orientation in (k,phi) form the IMU
    std::sprintf(text, "IMU Torso (k,phi): (%.3f, %.3f, %.3f, %.3f)", kphi[0], kphi[1], kphi[2], kphi[3]);
    mjr_label(vp_text, mjFONT_NORMAL, text, bgColor[0], bgColor[1], bgColor[2], bgColor[3], textColor[0], textColor[1], textColor[2], &con);

    // Move to the next line and adjust the viewport
    vp_text.bottom -= static_cast<int>(vp_text.height * lineSpacing);

    // Calculate (k,phi) from the quaternion
    my_mju_quat2axisAngle(readIn_for_rendering1->lf_quat, kphi);

    // Format and render the left foot orientation in (k,phi) form the IMU
    std::sprintf(text, "IMU LF (k,phi): (%.3f, %.3f, %.3f, %.3f)", kphi[0], kphi[1], kphi[2], kphi[3]);
    mjr_label(vp_text, mjFONT_NORMAL, text, bgColor[0], bgColor[1], bgColor[2], bgColor[3], textColor[0], textColor[1], textColor[2], &con);

    // Move to the next line and adjust the viewport
    vp_text.bottom -= static_cast<int>(vp_text.height * lineSpacing);

    // Calculate (k,phi) from the quaternion
    my_mju_quat2axisAngle(readIn_for_rendering1->rf_quat, kphi);

    // Format and render the right foot orientation in (k,phi) form the IMU
    std::sprintf(text, "IMU RF (k,phi): (%.3f, %.3f, %.3f, %.3f)", kphi[0], kphi[1], kphi[2], kphi[3]);
    mjr_label(vp_text, mjFONT_NORMAL, text, bgColor[0], bgColor[1], bgColor[2], bgColor[3], textColor[0], textColor[1], textColor[2], &con);

    /********************************************************************************************************************************************/

    // Move to the next line and adjust the viewport
    vp_text.bottom -= static_cast<int>(vp_text.height * lineSpacing);

    // Format and render the torso angular velocity from IMU
    std::sprintf(text, "IMU Torso Ang Vel: (%.3f, %.3f, %.3f)", readIn_for_rendering1->torso_w[0], readIn_for_rendering1->torso_w[1], readIn_for_rendering1->torso_w[2]);
    mjr_label(vp_text, mjFONT_NORMAL, text, bgColor[0], bgColor[1], bgColor[2], bgColor[3], textColor[0], textColor[1], textColor[2], &con);

    // Move to the next line and adjust the viewport
    vp_text.bottom -= static_cast<int>(vp_text.height * lineSpacing);

    // Format and render the left foot angular velocity from IMU
    std::sprintf(text, "IMU LF Ang Vel: (%.3f, %.3f, %.3f)", readIn_for_rendering1->lf_w[0], readIn_for_rendering1->lf_w[1], readIn_for_rendering1->lf_w[2]);
    mjr_label(vp_text, mjFONT_NORMAL, text, bgColor[0], bgColor[1], bgColor[2], bgColor[3], textColor[0], textColor[1], textColor[2], &con);

    // Move to the next line and adjust the viewport
    vp_text.bottom -= static_cast<int>(vp_text.height * lineSpacing);

    // Format and render the right foot angular velocity from IMU
    std::sprintf(text, "IMU RF Ang Vel: (%.3f, %.3f, %.3f)", readIn_for_rendering1->rf_w[0], readIn_for_rendering1->rf_w[1], readIn_for_rendering1->rf_w[2]);
    mjr_label(vp_text, mjFONT_NORMAL, text, bgColor[0], bgColor[1], bgColor[2], bgColor[3], textColor[0], textColor[1], textColor[2], &con);

    /********************************************************************************************************************************************/
    
}

int main(int argc, const char **argv)
{
      // check command-line arguments
  if (argc != 2)
  {
    std::printf(" USAGE:  basic modelfile\n");
    return 0;
  }

  // load and compile model
  char error[1000] = "Could not load binary model";
  if (std::strlen(argv[1]) > 4 && !std::strcmp(argv[1] + std::strlen(argv[1]) - 4, ".mjb"))
  {
    m = mj_loadModel(argv[1], 0);
  }
  else
  {
    m = mj_loadXML(argv[1], 0, error, 1000);
  }
  if (!m)
  {
    mju_error("Load model error: %s", error);
  }

    // d_sim is Global and modified only by the simulation thread
    d_sim = mj_makeData(m);
    // d_exp is Global and modified only by the comm_with_robot thread.
    d_exp = mj_makeData(m);

    // The robot states are shared memory locations.
    unsigned int my_spec = mjSTATE_TIME | mjSTATE_QPOS | mjSTATE_QVEL | mjSTATE_CTRL | mjSTATE_USERDATA;
    int nStateVar = mj_stateSize(m, my_spec);
    printf("State size: %d\n", nStateVar);

    // Allocate mjtNum array of nStateVar size
    mjtNum *state_sim_for_rendering = (mjtNum *)mju_malloc(nStateVar * sizeof(mjtNum));
    mjtNum *state_sim_for_comm = (mjtNum *)mju_malloc(nStateVar * sizeof(mjtNum));
    mjtNum *state_exp = (mjtNum *)mju_malloc(nStateVar * sizeof(mjtNum));

    // A copy of model for the rendering thread.
    mjModel *m_ren = mj_copyModel(m_ren, m);

    // Copy of mjData for the rendering thread.
    mjData *d_sim_for_ren = mj_makeData(m);
    mjData *d_exp_for_ren = mj_makeData(m);
    ContactDataAnalysis mycda_sim_for_ren;
    RobotDataToRead readIn_for_ren;
    RobotDataToSend writeOut_for_ren;

    // Homing the force sensors
    communication_initialisation(fsr_noLoad_today, vRef_today);

    if (!glfwInit()) // initialise GLFW
    {
        mju_error("Could not initialize GLFW");
    }
    GLFWwindow *window = glfwCreateWindow(1920, 1080, "CPS", NULL, NULL); // Create window, make OpenGL context current, request v-sync
    my_glfw_initialisations(window);                                      // GLFW initialisations and callback setups
    my_visuals_initialisations();                                         // MuJoCo visualisation initialisations

    // Set rendering and comm states to the initial state.
    rendering_over.clear();
    comm_over.clear();
    unsigned int tmp;

    // Launch the threads:
    std::thread simulation_thread(simulation, state_sim_for_rendering, state_sim_for_comm);
    std::this_thread::sleep_for(std::chrono::milliseconds(10)); // Wait for 10 ms
    std::thread comm_robot_thread(communication_with_robot, state_sim_for_comm, state_exp);

    // Rendering at 60 Hz (16.67 ms), and event loop
    while (!glfwWindowShouldClose(window))
    {
        // Copy the simulated robot state to be rendered. Wait on the condition variable.
        std::unique_lock<std::mutex> sim_ren_lck(mtx_state_sim_for_rendering);
        cv_rendering.wait_for(sim_ren_lck, std::chrono::milliseconds(8));
        mj_setState(m_ren, d_sim_for_ren, state_sim_for_rendering, my_spec);
        sim_ren_lck.unlock();

        // Copy the experiment robot state to be rendered. This input is at lower frequency.
        mtx_state_exp.try_lock_for(std::chrono::milliseconds(3));
        mj_setState(m_ren, d_exp_for_ren, state_exp, my_spec);
        mtx_state_exp.unlock();

        // update data structures.
        mj_forward(m_ren, d_sim_for_ren);
        copyCDAFromDoubleArray(&mycda_sim_for_ren, d_sim_for_ren->userdata + ADDR_CDA);
        memcpy(&readIn_for_ren, d_exp_for_ren->userdata + ADDR_ROBOT_DATA_TO_READ, sizeof(RobotDataToRead));
        memcpy(&writeOut_for_ren, d_exp_for_ren->userdata + ADDR_ROBOT_DATA_TO_SEND, sizeof(RobotDataToSend));
        mjv_updateScene(m_ren, d_sim_for_ren, &vOpt, NULL, &cam, mjCAT_ALL, &scn_sim);
        mj_forward(m_ren, d_exp_for_ren);
        mjv_updateScene(m_ren, d_exp_for_ren, &vOpt, NULL, &cam, mjCAT_ALL, &scn_exp);

        // update other visualisations and figures (relative timing not exact in these plots)
        update_qComp_figures(d_sim_for_ren->userdata + ADDR_ROBOT_DESIRED_Q, d_sim_for_ren->qpos + 7, d_exp_for_ren->qpos + 7);
        double q_err_sim[12], q_err_exp[12];
        mju_sub(q_err_sim, d_sim_for_ren->userdata + ADDR_ROBOT_DESIRED_Q, d_sim_for_ren->qpos + 7, 12);
        mju_sub(q_err_exp, d_sim_for_ren->userdata + ADDR_ROBOT_DESIRED_Q, d_exp_for_ren->qpos + 7, 12);
        double deg2rad = M_PI / 180;
        mju_scl(q_err_sim, q_err_sim, 180 / M_PI, 12); // Convert to degrees
        mju_scl(q_err_exp, q_err_exp, 180 / M_PI, 12); // Convert to degrees
        update_qErr_figures(q_err_sim, q_err_exp);
        update_sim_COP_plot(&fig_gcz_trace_sim, m_ren, d_sim_for_ren, &mycda_sim_for_ren);
        update_exp_COP_plot(&fig_gcz_trace_exp, m_ren, d_sim_for_ren, &mycda_sim_for_ren, &readIn_for_ren);

        // get framebuffer viewport
        mjrRect vp_full = {0, 0, 0, 0};
        glfwGetFramebufferSize(window, &vp_full.width, &vp_full.height);
        mjrRect vp_sim = {0, 0, 0, 0}, vp_exp = {0, 0, 0, 0};
        // Left top corner
        vp_sim.width = vp_full.width / 2;
        vp_sim.height = vp_full.height / 2;
        vp_sim.left = 0;
        vp_sim.bottom = vp_full.height / 2;
        // Left bottom corner
        vp_exp.width = vp_full.width / 2;
        vp_exp.height = vp_full.height / 2;
        vp_exp.left = 0;
        vp_exp.bottom = 0;

        // Render the scene
        mjr_render(vp_sim, &scn_sim, &con);
        mjr_render(vp_exp, &scn_exp, &con);

        // Render other visualisations and figures
        render_qComp_figures(vp_full);
        render_qErr_figures(vp_full);
        render_sim_COP_plot(vp_full);
        render_exp_COP_plot(vp_full);

        // Render text
        render_text(m_ren, d_sim_for_ren, d_exp_for_ren, &readIn_for_ren, vp_full);

        // swap OpenGL buffers (blocking call due to v-sync)
        glfwSwapBuffers(window);
        // notedown the rendering time.
        tmp = static_cast<unsigned int>(10000.0 * glfwGetTime());
        time_rendering.store(tmp, std::memory_order_release);
        rendering_over.clear();

        // Print d->time, glfwGetTime(), and the difference
        // double curr_time = glfwGetTime();
        // printf("Ren d_time: %.4f, glfwTime: %.4f, D: %.4f\n", d_sim_for_ren->time, curr_time, curr_time - d_sim_for_ren->time);

        // process pending GUI events, call GLFW callbacks
        glfwPollEvents();
    }

    // Termination: free up the resources and close the communication threads.
    start_stop_comm = 0;
    // Sleep for 100 ms to allow the threads to finish.
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    comm_robot_thread.join();
    simulation_thread.join();

    // free visualization storage
    mjv_freeScene(&scn_sim);
    mjv_freeScene(&scn_exp);
    mjr_freeContext(&con);

    // free MuJoCo model and data
    mju_free(state_sim_for_rendering);
    mju_free(state_sim_for_comm);
    mju_free(state_exp);

    mj_deleteData(d_sim_for_ren);
    mj_deleteData(d_exp_for_ren);
    mj_deleteData(d_sim);
    mj_deleteData(d_exp);

    mj_deleteModel(m);
    mj_deleteModel(m_ren);

    return 0;
}

// Simulation thread
void simulation(mjtNum *StateSimForRendering, mjtNum *StateSimSendToRobot)
{
    // Sim state vector spec
    unsigned int my_spec = mjSTATE_TIME | mjSTATE_QPOS | mjSTATE_QVEL | mjSTATE_CTRL | mjSTATE_USERDATA;
    double target_rendering = 0.0;
    double target_comm = 0.0;
    bool update_rendering_target = false;
    bool update_comm_target = false;
    unsigned int tmp;

    // Initialise the MuJoCo robot state from a file
    load_qpos(m, d_sim);
    load_sitepose(txyz, tkphi, lfxyz, lfkphi, rfxyz, rfkphi);
    // estimate for warmstarting IK solver
    mju_copy(qjans_hold, d_sim->qpos + 7, 12);
    mju_zero(dqjans_hold, 12);
    double nom_speed[12] = {3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3};
    mju_addTo(dqjans, nom_speed, 12);
    mju_copy(qjans, qjans_hold, 12);
    mju_copy(dqjans, dqjans_hold, 12);

    // Copy the state for communication for the first iteration of communication
    {
        // copyCDAToDoubleArray(d_sim->userdata + ADDR_CDA, &mycda_sim);
        std::lock_guard<std::mutex> lock(mtx_state_sim_for_comm);
        mj_getState(m, d_sim, StateSimSendToRobot, my_spec);
    }
    update_comm_target = false;
    cv_comm.notify_all();

    // Copy state for rendering for the first iteration of rendering
    {
        // copyCDAToDoubleArray(d_sim->userdata + ADDR_CDA, &mycda_sim);
        std::lock_guard<std::mutex> lock(mtx_state_sim_for_rendering);
        mj_getState(m, d_sim, StateSimForRendering, my_spec);
    }
    update_rendering_target = false;
    cv_rendering.notify_all();

    // Simulate
    while (start_stop_comm)
    {

        if (ppause)
        {
            d_sim->time = glfwGetTime();
            std::this_thread::sleep_for(std::chrono::milliseconds(4));
            continue;
        }

        // Update rendering target time?
        if (!rendering_over.test_and_set(std::memory_order_acquire))
            update_rendering_target = true;

        // Update comm target time?
        if (!comm_over.test_and_set(std::memory_order_acquire))
            update_comm_target = true;

        // If no one needs the simulation to run, sleep for 500 us
        if (!(update_rendering_target || update_comm_target))
        {
            // Sleep for 500 us
            std::this_thread::sleep_for(std::chrono::microseconds(2000));
        }

        // If only rendering needs simulation
        if (update_rendering_target)
        {
            {
                std::lock_guard<std::timed_mutex> lk(mtx_d_sim);
                pose_updater(m, d_sim);
                mj_forward(m, d_sim);
                updateContactDataAnalysis(m, d_sim, &mycda_sim);
            }
            // Copy the state for rendering
            {
                mju_copy(d_sim->userdata + ADDR_ROBOT_DESIRED_Q, qjans, 12);
                mju_copy(d_sim->userdata + ADDR_ROBOT_DESIRED_QDOT + 12, dqjans, 12);
                copyCDAToDoubleArray(d_sim->userdata + ADDR_CDA, &mycda_sim);
                std::lock_guard<std::mutex> lock(mtx_state_sim_for_rendering);
                mj_getState(m, d_sim, StateSimForRendering, my_spec);
            }
            update_rendering_target = false;
            cv_rendering.notify_all();
        }

        // If only communication needs simulation
        if (update_comm_target)
        {
            {
                std::lock_guard<std::timed_mutex> lk(mtx_d_sim);
                pose_updater(m, d_sim);
                mj_forward(m, d_sim);
                updateContactDataAnalysis(m, d_sim, &mycda_sim);
            }
            // Copy the state for communication
            {
                mju_copy(d_sim->userdata + ADDR_ROBOT_DESIRED_Q, qjans, 12);
                mju_copy(d_sim->userdata + ADDR_ROBOT_DESIRED_QDOT + 12, dqjans, 12);
                copyCDAToDoubleArray(d_sim->userdata + ADDR_CDA, &mycda_sim);
                std::lock_guard<std::mutex> lock(mtx_state_sim_for_comm);
                mj_getState(m, d_sim, StateSimSendToRobot, my_spec);
            }
            update_comm_target = false;
            cv_comm.notify_all();
        }
    }
}

/**************************************************************************************/
// Initialise the communication to the robot. Force stick support suspended at the moment.
void communication_initialisation(int fsr_fresh[], int vref_fresh[])
{
    int calibration_initialisation = 0;
    while (1)
    {
        printf("Skip calibration? Press 'y' to skip, presss Enter to continue.\n");
        calibration_initialisation = getchar();
        if (calibration_initialisation == 121)
        {
            vref_fresh[0] = 212;
            vref_fresh[1] = 212;
            vref_fresh[2] = 212;
            vref_fresh[3] = 212; // End left foot.
            vref_fresh[4] = 206;
            vref_fresh[5] = 206;
            vref_fresh[6] = 206;
            vref_fresh[7] = 206; // End Right foot.
            vref_fresh[8] = 224; // FS
            printf("vRef_assumed: %d %d %d %d %d %d %d %d %d\n", vref_fresh[0],
                   vref_fresh[1], vref_fresh[2], vref_fresh[3], vref_fresh[4],
                   vref_fresh[5], vref_fresh[6], vref_fresh[7], vref_fresh[8]);
            memcpy(fsr_fresh, vref_fresh, 8 * sizeof(int));
            printf("Assuming zero preload.\n");
            break;
        }
        else
        {
            communication_initialisation_robot(fsr_fresh, vref_fresh);
            // communication_initialisation_forcestick(fsr_fresh, vref_fresh);
            printf("vRef_today: %d %d %d %d %d %d %d %d %d\n", vref_fresh[0],
                   vref_fresh[1], vref_fresh[2], vref_fresh[3], vref_fresh[4],
                   vref_fresh[5], vref_fresh[6], vref_fresh[7], vref_fresh[8]);
            printf("\nSatisfied with the calibration? Press Enter to continue. Press "
                   "any other key to redo the calibration.\n");
            calibration_initialisation = getchar();
            if (calibration_initialisation == 10)
                break;
        }
    }
}

// This function adjusts the force sensor preload that is used to compute the
// ground reaction readings.
void communication_initialisation_robot(int fsr_today[], int vref_today[])
{
    // Initiate the communication to the robot and get the Vref reading along with
    // the preload values.
    SerialTransfer::SerialTransfer initIO("/dev/ttyUSB0", 1000000);

    // Define call packets to be sent to the microcontrollers on the robot: {From,
    // To, Function Code}
    uint8_t packet_send[3], size_packet_send = 3;

    // Define the packet to receive data from the robot.
    uint8_t packet_received[64], size_packet_received = 0;
    int s[4], vref;

    // Build the packet to send to the left foot.
    packet_send[0] = PC_add;            // From
    packet_send[1] = L_add;             // To
    packet_send[2] = Req_fsr_init_data; // Function code

    printf("Hold the robot in the air and press Enter. Noting the readings of the left foot.\n");
    getchar();
    for (int i = 0; i < 10; i++)
    {
        // Flush the serial port.
        initIO.reset();
        // Broadcast the packet.
        printf("Sending packet to the left foot.\n");
        initIO.txObj(packet_send, 0,
                     size_packet_send);    // Load the packet in the Tx buffer
        initIO.sendData(size_packet_send); // Send the packet.
        usleep(DATA_OUT_SLEEP);            // Sleep till the data is sent out.

        // Receive the packet from the left foot.
        size_packet_received = 0;
        // Note the time using chrono
        auto time_start = std::chrono::steady_clock::now();
        while (!size_packet_received)
        {
            if (initIO.available())
            {
                // Read the incomming packet.
                size_packet_received = initIO.bytesRead;
                initIO.rxObj(packet_received, 0, size_packet_received);
            }
            // Check for timeout.
            auto time_difference = std::chrono::duration_cast<std::chrono::microseconds>(
                                       std::chrono::steady_clock::now() - time_start)
                                       .count();
            if (time_difference > RS485_TO)
            {
                printf("Timeout. No packet received from the left foot.\n");
                break;
            }
        }
        // Convert an array of 2-byte (LB|HB) integers to an array of 4-byte
        // integers.
        s[0] = (packet_received[3] | packet_received[4] << 8);
        s[1] = (packet_received[5] | packet_received[6] << 8);
        s[2] = (packet_received[7] | packet_received[8] << 8);
        s[3] = (packet_received[9] | packet_received[10] << 8);
        vref = (packet_received[11] | packet_received[12] << 8);

        // Print the received data with its description.
        printf("Left foot: Vref:%d\tJ0:%d\tJ1:%d\tJ2:%d\tJ3:%d\n", vref, s[0], s[1], s[2], s[3]);
    }
    printf("Press Enter to record the Vref and preloaded sensor readings for the left foot.\n");
    getchar();

    // Copy the Vref and preloaded sensor readings for the left foot.
    for (int i = 0; i < 4; i++)
    {
        vref_today[i] = vref;
        fsr_today[i] = s[i];
    }

    /**********************************************************************************************/
    // Build the packet to send to the right foot.
    packet_send[0] = PC_add;            // From
    packet_send[1] = R_add;             // To
    packet_send[2] = Req_fsr_init_data; // Function code

    printf("Hold the robot in the air and press Enter. Noting the readings of the right foot. \n");
    getchar();
    for (int i = 0; i < 10; i++)
    {
        // Flush the serial port.
        initIO.reset();
        // Broadcast the packet.
        printf("Sending packet to the right foot.\n");
        initIO.txObj(packet_send, 0,
                     size_packet_send);    // Load the packet in the Tx buffer
        initIO.sendData(size_packet_send); // Send the packet.
        usleep(DATA_OUT_SLEEP);            // Sleep till the data is sent out.

        // Receive the packet from the right foot.
        size_packet_received = 0;

        // Note the time using chrono
        auto time_start = std::chrono::steady_clock::now();
        while (!size_packet_received)
        {
            // hi2();
            if (initIO.available())
            {
                // Read the incomming packet.
                size_packet_received = initIO.bytesRead;
                initIO.rxObj(packet_received, 0, size_packet_received);
            }
            // Check for timeout.
            auto time_difference = std::chrono::duration_cast<std::chrono::microseconds>(
                                       std::chrono::steady_clock::now() - time_start)
                                       .count();
            if (time_difference > RS485_TO)
            {
                printf("Timeout. No packet received from the right foot.\n");
                break;
            }
        }
        // Convert an array of 2-byte (LB|HB) integers to an array of 4-byte
        // integers.
        s[0] = (packet_received[3] | packet_received[4] << 8);
        s[1] = (packet_received[5] | packet_received[6] << 8);
        s[2] = (packet_received[7] | packet_received[8] << 8);
        s[3] = (packet_received[9] | packet_received[10] << 8);
        vref = (packet_received[11] | packet_received[12] << 8);

        // Print the received data with its description.
        printf("Right foot: Vref:%d\tJ0:%d\tJ1:%d\tJ2:%d\tJ3:%d\n", vref, s[0],
               s[1], s[2], s[3]);
    }
    printf("Press Enter to record the Vref and preloaded sensor readings for the right foot.\n");
    getchar();

    // Copy the Vref and preloaded sensor readings for the right foot.
    for (int i = 0; i < 4; i++)
    {
        vref_today[i + 4] = vref;
        fsr_today[i + 4] = s[i];
    }

    printf("Robot foot sensor data initialisation complete.\n");

    /**********************************************************************************************/

    // Get the robot to the home position

    // Initialise the MuJoCo robot state from a file
    load_qpos(m, d_sim);
    load_sitepose(txyz, tkphi, lfxyz, lfkphi, rfxyz, rfkphi);
    // estimate for warmstarting IK solver
    mju_copy(qjans_hold, d_sim->qpos + 7, 12);
    mju_zero(dqjans_hold, 12);
    mju_copy(qjans, qjans_hold, 12);
    mju_copy(dqjans, dqjans_hold, 12);
    struct RobotDataToSend writeOut;

    // Home the actual robot state based on the MuJoCo robot state.
    mju_copy(writeOut.q, qjans, 12);
    // Homing speed
    double tqd = 1;
    double trial_qdot[12] = {tqd, tqd, tqd, tqd, tqd, tqd, tqd, tqd, tqd, tqd, tqd, tqd};
    mju_copy(writeOut.qdot, trial_qdot, 12);

    uint8_t packet_send2[51], size_packet_send2 = 51; // 3 + 48(q12x2|qdot12x2) = 51
    // Build the packet to send to the robot.
    size_packet_send2 = packet_send_builder(packet_send2, &writeOut);
    // Flush the serial port.
    initIO.reset();
    // Load the packet in the Tx buffer
    initIO.txObj(packet_send2, 0, size_packet_send2);
    // Send the packet
    initIO.sendData(size_packet_send2);
    // Short sleep till the data is sent out.
    usleep(DATA_OUT_SLEEP);

    printf("Robot homing initiated.\n");

    printf("Press Enter to continue.\n");
    getchar();

    // initIO automatically desctructs as it goes out of scope.
}

/**************************************************************************************/

void communication_with_robot(mjtNum *StateSimSendToRobot, mjtNum *StateExp)
{
    // Open the COM port and initiate the communication.
    SerialTransfer::SerialTransfer myrobotIO("/dev/ttyUSB0", 1000000);

    // Copy of mjModel and mjData_sim for the communication thread.
    mjModel *m_comm = mj_copyModel(m_comm, m);
    unsigned int my_spec = mjSTATE_TIME | mjSTATE_QPOS | mjSTATE_QVEL | mjSTATE_CTRL | mjSTATE_USERDATA;
    mjData *d_sim_for_comm = mj_makeData(m_comm);

    // Initialise the RobotoDataToSend structure.
    RobotDataToSend writeOut;
    // Zero initialise the RobotDataToRead structure.
    RobotDataToRead readIn_filtered;
    RobotDataToRead readIn_raw = {};

    // Define call packets to be sent to the microcontrollers on the robot: {FromTo, Function Code}
    uint8_t packet_send[51], size_packet_send = 51; // 3 + 48(q12x2|qdot12x2) = 51
    // Define the packet to receive data from the robot.
    uint8_t packet_received[64], size_packet_received = 64;
    int replies = 0;

    // Filter window initialisation.
    std::deque<RobotDataToRead> queReadIn;
    // Fill it with dummy zero data of SIZE_FILTER_WINDOW
    for (int i = 0; i < SIZE_FILTER_WINDOW; i++)
    {
        queReadIn.push_back(readIn_raw);
    }

    // Timing variables
    auto loop_time_start = std::chrono::steady_clock::now();  // Initialise the loop timer.
    auto reply_time_start = std::chrono::steady_clock::now(); // Initialise the reply timer.
    auto sleep_time_start = std::chrono::steady_clock::now(); // Initialise the sleep timer.
    auto current_time = std::chrono::steady_clock::now();     // Initialise the current time.
    auto last_time = std::chrono::steady_clock::now();        // Initialise the last time.
    unsigned int tmp;
    printf("Robot communication thread started.\n");

    // Communication loop running at ~50Hz
    while (start_stop_comm)
    {
        if (ppause)
        {
            std::this_thread::sleep_for(std::chrono::milliseconds(4));
            continue;
        }

        // Reset the loop timer
        loop_time_start = std::chrono::steady_clock::now();

        // Copy the simulation state to be sent to the robot.
        std::unique_lock<std::mutex> sim_comm_lck(mtx_state_sim_for_comm);
        cv_comm.wait_for(sim_comm_lck, std::chrono::milliseconds(1));
        // cv_comm.wait(sim_comm_lck);
        mj_setState(m_comm, d_sim_for_comm, StateSimSendToRobot, my_spec);
        sim_comm_lck.unlock();

        mj_forward(m_comm, d_sim_for_comm);
        mju_copy(writeOut.q, (d_sim_for_comm->qpos + 7), 12);
        double nom_speed[12] = {3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3};
        mju_copy(writeOut.qdot, nom_speed, 12);

        // Build the packet to send to the robot.
        size_packet_send = packet_send_builder(packet_send, &writeOut);
        // Flush the serial port.
        myrobotIO.reset();
        // Load the packet in the Tx buffer
        myrobotIO.txObj(packet_send, 0, size_packet_send);
        // Send (broadcast) the packet
        myrobotIO.sendData(size_packet_send);
        // Short sleep till the data is sent out.
        std::this_thread::sleep_for(std::chrono::microseconds(DATA_OUT_SLEEP));
        // Note down the time when data was sent to the robot.
        tmp = static_cast<unsigned int>(10000 * glfwGetTime());
        time_comm.store(tmp, std::memory_order_release);
        comm_over.clear(std::memory_order_release);

        // Print d->time, glfwGetTime(), and the difference
        // double curr_time = glfwGetTime();
        // printf("Comm d_time: %.4f, glfwTime: %.4f, D: %.4f\n", d_sim_for_comm->time, curr_time, curr_time - d_sim_for_comm->time);
        // Print qpos only LL
        // mju_printMat(d_sim_for_comm->qvel + 6, 1, 6);
        // mju_printMat(writeOut.qdot, 1, 12);

        // Read the incomming data from the robot.
        replies = 0;
        reply_time_start = std::chrono::steady_clock::now(); // Reset the reply timer.
        while (replies < 4)
        {
            if (myrobotIO.available()) // Read the incomming packet and process it.
            {
                size_packet_received = myrobotIO.bytesRead;
                myrobotIO.rxObj(packet_received, 0, size_packet_received);

                // Report time taken for the reply.
                // current_time = std::chrono::steady_clock::now();
                // auto td = std::chrono::duration_cast<std::chrono::microseconds>(current_time - reply_time_start).count();
                // std::cout << "RT: " << td << std::endl;

                // Process the incomming packet.
                packet_received_analyser(packet_received, size_packet_received, &readIn_raw);

                // Increment the reply counter.
                replies++;

                if (replies == 4) // Start SLEEP_BETWEEN_CALLS, apply filter, and break the loop.
                {
                    // printf("All in.\n");

                    // Store the fresh data in the queue.
                    queReadIn.push_back(readIn_raw);
                    // Remove the oldest data.
                    queReadIn.pop_front();

                    // Apply filter to the data queue.
                    if (filter_flag)
                    {
                        // Apply the filter to the data queue.
                        filter_robotData(queReadIn, &readIn_filtered);
                    }
                    else
                    {
                        // Use the raw data as it is
                        readIn_filtered = readIn_raw;
                    }

                    break;
                }
            }

            // Check for timeout.
            current_time = std::chrono::steady_clock::now();
            auto td_TO = std::chrono::duration_cast<std::chrono::microseconds>(current_time - reply_time_start).count();
            if ((td_TO) > (int64_t(RS485_TO))) // Timeout. Break the loop. Start SLEEP_BETWEEN_CALLS.
            {
                printf("TO: %d at time: %ld\n", replies, td_TO);
                break;
            }
        }

        // Robot reconstruction
        robot_reconstruction(recons_num, m_comm, d_sim_for_comm, d_exp, &readIn_filtered);

        // Copy the readIn data + WriteOut data to d_exp->userdata
        memcpy(d_exp->userdata + ADDR_ROBOT_DATA_TO_READ, &readIn_filtered, sizeof(RobotDataToRead));
        memcpy(d_exp->userdata + ADDR_ROBOT_DATA_TO_SEND, &writeOut, sizeof(RobotDataToSend));
        // Update the StateExp
        mtx_state_exp.lock();
        mj_getState(m_comm, d_exp, StateExp, my_spec);
        mtx_state_exp.unlock();

        // print the time since last loop
        // current_time = std::chrono::steady_clock::now();
        // double loop_time = std::chrono::duration_cast<std::chrono::microseconds>(current_time - last_time).count();
        // printf("Comm loop ms: %.3f\n", loop_time / 1000.0);
        // last_time = current_time;

        // Sleep till LOOP_TIME is over.
        current_time = std::chrono::steady_clock::now();
        while (std::chrono::duration_cast<std::chrono::microseconds>(current_time - loop_time_start).count() < int64_t(LOOP_TIME))
        {
            std::this_thread::sleep_for(std::chrono::microseconds(100));
            current_time = std::chrono::steady_clock::now();
        }

        // Compute the time taken for the loop.
        // current_time = std::chrono::steady_clock::now();
        // auto loop_time = std::chrono::duration_cast<std::chrono::microseconds>(current_time - loop_time_start).count();
        // // Print the loop time
        // std::cout << "Loop time: " << loop_time << std::endl;
    }

    // Free resources
    mj_deleteData(d_sim_for_comm);
    mj_deleteModel(m_comm);
}

void filter_robotData(std::deque<RobotDataToRead> &queReadIn, RobotDataToRead *readIn)
{
    /*
    The input double ended queue is pased by value because we do not want the original deque to get affected.
    The filter is applied to each element of q and qdot across the queue.
    The filter is applied to each element of torso gyro, lf_gyro, and rf_gyro.
    The median value is taken as the filtered value.
    The filtered value is stored in the readIn structure.
    */

    std::array<double, SIZE_FILTER_WINDOW> qvals;
    std::array<double, SIZE_FILTER_WINDOW> qdotvals;
    std::array<double, SIZE_FILTER_WINDOW> tw, lfw, rfw;

    // Copy all the fresh data from the latest item in deque to readIn_filtered.
    if (!queReadIn.empty())
    {
        *readIn = queReadIn.back();
    }

    // Apply filter to q and qdot
    for (int i = 0; i < 12; i++)
    {
        // Loop through the deque to fill qvals and qdotvals.
        for (int j = 0; j < SIZE_FILTER_WINDOW; j++)
        {
            qvals[j] = queReadIn[j].q[i];
            qdotvals[j] = queReadIn[j].qdot[i];
        }

        // Sort qvals and qdotvals.
        std::sort(qvals.begin(), qvals.end());
        std::sort(qdotvals.begin(), qdotvals.end());

        // Find the median value and assign it to readIn.
        readIn->q[i] = qvals[SIZE_FILTER_WINDOW / 2];
        readIn->qdot[i] = qdotvals[SIZE_FILTER_WINDOW / 2];
    }

    // Apply filter to torso, lf, and rf gyro values
    for (int i = 0; i < 3; i++)
    {
        // Loop through the deque to fill tw, lfw, and rfw.
        for (int j = 0; j < SIZE_FILTER_WINDOW; j++)
        {
            tw[j] = queReadIn[j].torso_w[i];
            lfw[j] = queReadIn[j].lf_w[i];
            rfw[j] = queReadIn[j].rf_w[i];
        }

        // Sort tw, lfw, and rfw.
        std::sort(tw.begin(), tw.end());
        std::sort(lfw.begin(), lfw.end());
        std::sort(rfw.begin(), rfw.end());

        // Find the median value and assign it to readIn.
        readIn->torso_w[i] = tw[SIZE_FILTER_WINDOW / 2];
        readIn->lf_w[i] = lfw[SIZE_FILTER_WINDOW / 2];
        readIn->rf_w[i] = rfw[SIZE_FILTER_WINDOW / 2];
    }
}

void robot_reconstruction(int num, mjModel *mm, mjData *dSim, mjData *dExp, RobotDataToRead *read_in)
{

    if (num == 0 || num == 1)
    {
        // Reconstruction using fixed base R/L foot assumption.
        double Rb[9], bf[3], Rf[9], of[3], Rbf[9], sim_of_avg[3];
        int idx = (1 - num) * LF_SITE + num * RF_SITE;

        // Computations in the frame of the floating base
        mj_resetData(mm, dExp);                              // Reset the dExp
        mju_copy(dExp->qpos + 7, read_in->q, 12);            // Using only the joint encoder data to reconstruct the robot pose
        mj_kinematics(mm, dExp);                             // Run forward kinematics: updates xpos, xmat, site_xpos, site_xmat
        mju_sub3(bf, dExp->site_xpos + 3 * idx, dExp->qpos); // LF/RF site - Floating base position
        mju_copy(Rbf, dExp->site_xmat + 9 * idx, 9);         // LF/RF site orientation w.r.t. the floating base

        // Solve for the robot floating base absolute position
        mju_mulMatMatT(Rb, dSim->site_xmat + 9 * idx, Rbf, 3, 3, 3); // R0b = R0f*Rbf'
        mju_copy3(of, dSim->site_xpos + 3 * idx);                    // LF/RF site absolute position
        mju_scl3(bf, bf, -1);                                        // -bf
        mju_mulMatVec(dExp->qpos, Rb, bf, 3, 3);                     // ob = -R0b*bf
        mju_add3(dExp->qpos, dExp->qpos, of);                        // ob = ob + of
        mju_mat2Quat(dExp->qpos + 3, Rb);                            // qob = R2q(R0b)
        mj_forward(mm, dExp);                                        // Update the model to get absolute pose
        dExp->time = dSim->time + 0.006;                             // It takes around 10-12 ms to receieve all encoder data
    }
    else if (num == 2)
    {
        // Reconstruction using avg foot base assumption.
        double Rb[9], bf[3], Rf[9], of[3], Rbf[9], sim_of_avg[3];

        // Computations in the frame of the floating base
        mj_resetData(mm, dExp);                                                // Reset the dExp
        mju_copy(dExp->qpos + 7, read_in->q, 12);                              // Using only the joint encoder data to reconstruct the robot pose
        mj_kinematics(mm, dExp);                                               // Run forward kinematics: updates xpos, xmat, site_xpos, site_xmat
        mju_add3(bf, dExp->site_xpos + 3, dExp->site_xpos + 6);                // LF+RF site
        mju_scl3(bf, bf, 0.5);                                                 // LF+RF site avg
        mju_sub3(bf, bf, dExp->qpos);                                          // LF+RF site - Floating base position
        my_mju_avgRmat(dExp->site_xmat + 9 * 1, dExp->site_xmat + 9 * 2, Rbf); // LF+RF site avg orientation

        // Solve for the robot floating base absolute position
        my_mju_avgRmat(dSim->site_xmat + 9 * 1, dSim->site_xmat + 9 * 2, Rf); // LF+RF site avg orientation of simulated robot
        mju_mulMatMatT(Rb, Rf, Rbf, 3, 3, 3);                                 // R0b = R0f*Rbf'
        mju_add3(sim_of_avg, dSim->site_xpos + 3, dSim->site_xpos + 6);       // LF+RF site avg absolute position of simulated robot
        mju_scl3(of, sim_of_avg, 0.5);                                        // LF+RF site avg absolute position of simulated robot
        mju_scl3(bf, bf, -1);                                                 // -bf
        mju_mulMatVec(dExp->qpos, Rb, bf, 3, 3);                              // ob = -R0b*bf
        mju_add3(dExp->qpos, dExp->qpos, of);                                 // ob = ob + of
        mju_mat2Quat(dExp->qpos + 3, Rb);                                     // qob = R2q(R0b)
        mj_forward(mm, dExp);                                                 // Update the model to get absolute pose
        dExp->time = dSim->time + 0.006;                                      // It takes around 10-12 ms to receieve all encoder data
    }
}

/**************************************************************************************/

void pose_updater(const mjModel *mm, mjData *dd)
{
    /*     mju_printMat(txyz, 1, 3);
        mju_printMat(lfxyz, 1, 3);
        mju_printMat(rfxyz, 1, 3);
        printf("\n"); */
    // bioloid_12dof_IK_position_v2(mm, TORSO_SITE, LF_SITE, RF_SITE, txyz, tkphi, lfxyz, lfkphi, rfxyz, rfkphi, qjans, 20, 1e-6, 1e-5);
    bioloid_12dof_IK_position_v3(mm, TORSO_SITE, LF_SITE, RF_SITE, txyz, tkphi, lfxyz, lfkphi, rfxyz, rfkphi, qjans, 50, 1e-6, 1e-5);

    // qjans is updated only if solution converges

    // update the pose--------------------------------------------------
    double global_torso_quat[4], global_leftfoot_quat[4], global_rightfoot_quat[4];
    mju_normalize3(tkphi);
    mju_normalize3(lfkphi);
    mju_normalize3(rfkphi);
    mju_axisAngle2Quat(global_torso_quat, tkphi, tkphi[3]);
    mju_axisAngle2Quat(global_leftfoot_quat, lfkphi, lfkphi[3]);
    mju_axisAngle2Quat(global_rightfoot_quat, rfkphi, rfkphi[3]);
    // Position of the site w.r.t the floating base
    int tsno = 0; // torso site id
    double torso_site_offsetpos[3], torso_site_offsetquat[4];
    mju_copy3(torso_site_offsetpos, mm->site_pos + tsno * 3);   // Correct
    mju_copy4(torso_site_offsetquat, mm->site_quat + tsno * 4); // Correct

    // Floating base orientation
    my_mju_relQuat(dd->qpos + 3, global_torso_quat, torso_site_offsetquat); // Correct
    // Floating base position
    double rot_fb[9];
    mju_quat2Mat(rot_fb, dd->qpos + 3);
    mju_mulMatVec(dd->qpos, rot_fb, torso_site_offsetpos, 3, 3);
    mju_sub3(dd->qpos, txyz, dd->qpos);
    //  Rest of the joints qj = [qjL|qjR]
    mju_copy(dd->qpos + 7, qjans, 12);

    // For updating control torques and contact forces
    // mju_copy(dd->ctrl, qjans, 12);
}

/**************************************************************************************/
// keyboard callback function, it uses global structure variables
void keyboard_cps(GLFWwindow *window, int key, int scancode, int act, int mods)
{
    double quat_temp[4], quat_current[4], xaxis[3] = {1, 0, 0}, yaxis[3] = {0, 1, 0}, zaxis[3] = {0, 0, 1}; // For the rotations
    if (act == GLFW_PRESS)
    {
        std::lock_guard<std::timed_mutex> lk(mtx_d_sim);
        switch (key)
        {
        case GLFW_KEY_1:
            // Choose torso site
            tf = true;
            lf = false;
            rf = false;
            printf("Torso site selected\n");
            break;

        case GLFW_KEY_2:
            // Choose left foot site
            tf = false;
            lf = true;
            rf = false;
            printf("Left foot site selected\n");
            break;

        case GLFW_KEY_3:
            // Choose right foot site
            tf = false;
            lf = false;
            rf = true;
            printf("Right foot sire selected\n");
            break;

        case GLFW_KEY_UP:
            // Y coord ++
            (tf && (txyz[1] += incr));
            (lf && (lfxyz[1] += incr));
            (rf && (rfxyz[1] += incr));
            break;

        case GLFW_KEY_DOWN:
            // Y coord --
            (tf && (txyz[1] -= incr));
            (lf && (lfxyz[1] -= incr));
            (rf && (rfxyz[1] -= incr));
            break;

        case GLFW_KEY_RIGHT:
            // X coord ++
            (tf && (txyz[0] += incr));
            (lf && (lfxyz[0] += incr));
            (rf && (rfxyz[0] += incr));
            break;

        case GLFW_KEY_LEFT:
            // X coord --
            (tf && (txyz[0] -= incr));
            (lf && (lfxyz[0] -= incr));
            (rf && (rfxyz[0] -= incr));
            break;

        case GLFW_KEY_PAGE_UP:
            // Z coord ++
            (tf && (txyz[2] += incr));
            (lf && (lfxyz[2] += incr));
            (rf && (rfxyz[2] += incr));
            break;

        case GLFW_KEY_PAGE_DOWN:
            // Z coord --
            (tf && (txyz[2] -= incr));
            (lf && (lfxyz[2] -= incr));
            (rf && (rfxyz[2] -= incr));
            break;

        case GLFW_KEY_4:
            // Pitch ccw rotation

            // Incremental rotation
            mju_axisAngle2Quat(quat_temp, xaxis, incr_ang);

            if (tf)
            {
                // get the quaternion corresponding to the current kphi
                mju_axisAngle2Quat(quat_current, tkphi, tkphi[3]);

                // mju_mulQuat(quat_temp, quat_temp, quat_current);
                // post multiply for body fixed rotations
                mju_mulQuat(quat_temp, quat_current, quat_temp);
                // set the total new kphi
                my_mju_quat2axisAngle(quat_temp, tkphi);
            }
            if (lf)
            {
                mju_axisAngle2Quat(quat_current, lfkphi, lfkphi[3]);
                // mju_mulQuat(quat_temp, quat_temp, quat_current);
                // post multiply for body fixed rotations
                mju_mulQuat(quat_temp, quat_current, quat_temp);
                my_mju_quat2axisAngle(quat_temp, lfkphi);
            }
            if (rf)
            {
                mju_axisAngle2Quat(quat_current, rfkphi, rfkphi[3]);
                // mju_mulQuat(quat_temp, quat_temp, quat_current);
                // post multiply for body fixed rotations
                mju_mulQuat(quat_temp, quat_current, quat_temp);
                my_mju_quat2axisAngle(quat_temp, rfkphi);
            }
            break;

        case GLFW_KEY_5:
            // Pitch cw rotation

            // Incremental rotation
            mju_axisAngle2Quat(quat_temp, xaxis, -incr_ang);

            if (tf)
            {
                // get the quaternion corresponding to the current kphi
                mju_axisAngle2Quat(quat_current, tkphi, tkphi[3]);
                // mju_mulQuat(quat_temp, quat_temp, quat_current);
                // post multiply for body fixed rotations
                mju_mulQuat(quat_temp, quat_current, quat_temp);
                // set the total new kphi
                my_mju_quat2axisAngle(quat_temp, tkphi);
            }
            if (lf)
            {
                mju_axisAngle2Quat(quat_current, lfkphi, lfkphi[3]);
                // mju_mulQuat(quat_temp, quat_temp, quat_current);
                // post multiply for body fixed rotations
                mju_mulQuat(quat_temp, quat_current, quat_temp);
                my_mju_quat2axisAngle(quat_temp, lfkphi);
            }
            if (rf)
            {
                mju_axisAngle2Quat(quat_current, rfkphi, rfkphi[3]);
                // mju_mulQuat(quat_temp, quat_temp, quat_current);
                // post multiply for body fixed rotations
                mju_mulQuat(quat_temp, quat_current, quat_temp);
                my_mju_quat2axisAngle(quat_temp, rfkphi);
            }

            break;
        case GLFW_KEY_6:
            // Roll ccw rotation

            // Incremental rotation
            mju_axisAngle2Quat(quat_temp, yaxis, incr_ang);

            if (tf)
            {
                // get the quaternion corresponding to the current kphi
                mju_axisAngle2Quat(quat_current, tkphi, tkphi[3]);
                // mju_mulQuat(quat_temp, quat_temp, quat_current);
                // post multiply for body fixed rotations
                mju_mulQuat(quat_temp, quat_current, quat_temp);
                // set the total new kphi
                my_mju_quat2axisAngle(quat_temp, tkphi);
            }
            if (lf)
            {
                mju_axisAngle2Quat(quat_current, lfkphi, lfkphi[3]);
                // mju_mulQuat(quat_temp, quat_temp, quat_current);
                // post multiply for body fixed rotations
                mju_mulQuat(quat_temp, quat_current, quat_temp);
                my_mju_quat2axisAngle(quat_temp, lfkphi);
            }
            if (rf)
            {
                mju_axisAngle2Quat(quat_current, rfkphi, rfkphi[3]);
                // mju_mulQuat(quat_temp, quat_temp, quat_current);
                // post multiply for body fixed rotations
                mju_mulQuat(quat_temp, quat_current, quat_temp);
                my_mju_quat2axisAngle(quat_temp, rfkphi);
            }
            break;

        case GLFW_KEY_7:
            // Roll cw rotation

            // Incremental rotation
            mju_axisAngle2Quat(quat_temp, yaxis, -incr_ang);

            if (tf)
            {
                // get the quaternion corresponding to the current kphi
                mju_axisAngle2Quat(quat_current, tkphi, tkphi[3]);
                // mju_mulQuat(quat_temp, quat_temp, quat_current);
                // post multiply for body fixed rotations
                mju_mulQuat(quat_temp, quat_current, quat_temp);
                // set the total new kphi
                my_mju_quat2axisAngle(quat_temp, tkphi);
            }
            if (lf)
            {
                mju_axisAngle2Quat(quat_current, lfkphi, lfkphi[3]);
                // mju_mulQuat(quat_temp, quat_temp, quat_current);
                // post multiply for body fixed rotations
                mju_mulQuat(quat_temp, quat_current, quat_temp);
                my_mju_quat2axisAngle(quat_temp, lfkphi);
            }
            if (rf)
            {
                mju_axisAngle2Quat(quat_current, rfkphi, rfkphi[3]);
                // mju_mulQuat(quat_temp, quat_temp, quat_current);
                // post multiply for body fixed rotations
                mju_mulQuat(quat_temp, quat_current, quat_temp);
                my_mju_quat2axisAngle(quat_temp, rfkphi);
            }
            break;
        case GLFW_KEY_8:
            // Yaw ccw rotation
            // Incremental rotation
            mju_axisAngle2Quat(quat_temp, zaxis, incr_ang);
            if (tf)
            {
                // get the quaternion corresponding to the current kphi
                mju_axisAngle2Quat(quat_current, tkphi, tkphi[3]);
                // mju_mulQuat(quat_temp, quat_temp, quat_current);
                // post multiply for body fixed rotations
                mju_mulQuat(quat_temp, quat_current, quat_temp);
                // set the total new kphi
                my_mju_quat2axisAngle(quat_temp, tkphi);
            }
            if (lf)
            {
                mju_axisAngle2Quat(quat_current, lfkphi, lfkphi[3]);
                // mju_mulQuat(quat_temp, quat_temp, quat_current);
                // post multiply for body fixed rotations
                mju_mulQuat(quat_temp, quat_current, quat_temp);
                my_mju_quat2axisAngle(quat_temp, lfkphi);
            }
            if (rf)
            {
                mju_axisAngle2Quat(quat_current, rfkphi, rfkphi[3]);
                // mju_mulQuat(quat_temp, quat_temp, quat_current);
                // post multiply for body fixed rotations
                mju_mulQuat(quat_temp, quat_current, quat_temp);
                my_mju_quat2axisAngle(quat_temp, rfkphi);
            }
            break;

        case GLFW_KEY_9:
            // Yaw cw rotation
            // Incremental rotation
            mju_axisAngle2Quat(quat_temp, zaxis, -incr_ang);
            if (tf)
            {
                // get the quaternion corresponding to the current kphi
                mju_axisAngle2Quat(quat_current, tkphi, tkphi[3]);
                // mju_mulQuat(quat_temp, quat_temp, quat_current);
                // post multiply for body fixed rotations
                mju_mulQuat(quat_temp, quat_current, quat_temp);
                // set the total new kphi
                my_mju_quat2axisAngle(quat_temp, tkphi);
            }
            if (lf)
            {
                mju_axisAngle2Quat(quat_current, lfkphi, lfkphi[3]);
                // mju_mulQuat(quat_temp, quat_temp, quat_current);
                // post multiply for body fixed rotations
                mju_mulQuat(quat_temp, quat_current, quat_temp);
                my_mju_quat2axisAngle(quat_temp, lfkphi);
            }
            if (rf)
            {
                mju_axisAngle2Quat(quat_current, rfkphi, rfkphi[3]);
                // mju_mulQuat(quat_temp, quat_temp, quat_current);
                // post multiply for body fixed rotations
                mju_mulQuat(quat_temp, quat_current, quat_temp);
                my_mju_quat2axisAngle(quat_temp, rfkphi);
            }
            break;

        case GLFW_KEY_G:
            // Toggle filter flag
            filter_flag ^= true;
            break;

        case GLFW_KEY_E:
            // Toggle through reconstruction modes
            recons_num = (recons_num + 1) % RECONS_Nmodes;
            break;

        case GLFW_KEY_RIGHT_ALT:
            // Toggle through figure pages
            fig_num = (fig_num + 1) % FIG_Npages;
            break;

        case GLFW_KEY_W:
            // Write qpos to file
            printf("Saving qpos to file: qpos_export.txt\n");
            save_qpos(m, d_sim);
            printf("Saving site frame data to site_export.txt\n");
            save_sitepose(txyz, tkphi, lfxyz, lfkphi, rfxyz, rfkphi);
            printf("Saving COM position to com_export.txt\n");
            save_com_pos(d_sim->subtree_com);
            break;

        case GLFW_KEY_R:
            // Read qpos from file
            printf("Loading qpos from file: qpos_export.txt\n");
            load_qpos(m, d_sim);
            printf("Loading site frame data from site_export.txt\n");
            load_sitepose(txyz, tkphi, lfxyz, lfkphi, rfxyz, rfkphi);
            break;

        case GLFW_KEY_F:
            // Contact force
            vOpt.flags[mjVIS_CONTACTFORCE] ^= true;
            break;

        case GLFW_KEY_T:
            // Transparency
            vOpt.flags[mjVIS_TRANSPARENT] ^= true;
            break;

        case GLFW_KEY_S:
            // Toggle through frames
            vOpt.frame = (vOpt.frame + 1) % mjNFRAME;
            break;

        case GLFW_KEY_J:
            vOpt.flags[mjVIS_JOINT] ^= true;
            break;

        case GLFW_KEY_L:
            // Toggle through labels
            vOpt.label = (vOpt.label + 1) % mjNLABEL;
            break;

        case GLFW_KEY_SPACE:
            // toggle ppause variable
            ppause ^= true;
            break;

        case GLFW_KEY_BACKSPACE:
            // backspace: reset simulation
            mj_resetData(m, d_sim);
            // load qpos0
            load_qpos(m, d_sim);

            // estimate for IK solver
            mju_copy(qjans, d_sim->qpos + 7, 12);
            mj_forward(m, d_sim);

            // Intentional fall through this case to the following case:
        case GLFW_KEY_ESCAPE:
            // Reset stuff
            
            break;
        }
    }
}

// mouse button callback
void mouse_button(GLFWwindow *window, int button, int act, int mods)
{
    // update button state
    button_left = (glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_LEFT) == GLFW_PRESS);
    button_middle = (glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_MIDDLE) == GLFW_PRESS);
    button_right = (glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_RIGHT) == GLFW_PRESS);

    // update mouse position
    glfwGetCursorPos(window, &lastx, &lasty);
}

// mouse move callback
void mouse_move(GLFWwindow *window, double xpos, double ypos)
{
    // no buttons down: nothing to do
    if (!button_left && !button_middle && !button_right)
    {
        return;
    }

    // compute mouse displacement, save
    double dx = xpos - lastx;
    double dy = ypos - lasty;
    lastx = xpos;
    lasty = ypos;

    // get current window size
    int width, height;
    glfwGetWindowSize(window, &width, &height);

    // get shift key state
    bool mod_shift = (glfwGetKey(window, GLFW_KEY_LEFT_SHIFT) == GLFW_PRESS || glfwGetKey(window, GLFW_KEY_RIGHT_SHIFT) == GLFW_PRESS);

    // determine action based on mouse button
    mjtMouse action;
    if (button_right)
    {
        action = mod_shift ? mjMOUSE_MOVE_H : mjMOUSE_MOVE_V;
    }
    else if (button_left)
    {
        action = mod_shift ? mjMOUSE_ROTATE_H : mjMOUSE_ROTATE_V;
    }
    else
    {
        action = mjMOUSE_ZOOM;
    }

    // move camera
    mjv_moveCamera(m, action, dx / height, dy / height, &scn_sim, &cam);
}

// scroll callback
void scroll(GLFWwindow *window, double xoffset, double yoffset)
{
    // emulate vertical mouse motion = 5% of window height
    mjv_moveCamera(m, mjMOUSE_ZOOM, 0, -0.05 * yoffset, &scn_sim, &cam);
}

/**************************************************************************************/
