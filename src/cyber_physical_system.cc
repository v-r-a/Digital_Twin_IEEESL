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
#include <ceres/ceres.h>

#include "../include/mujoco/mujoco.h"
#include "SerialTransfer.h"
#include "my_mjc_func_declarations.h"
#include "rs485_bus_functions.h"
#include "exp_meas_proc_func.h"

// Function declarations
void communication_initialisation(int fsr_fresh[], int vref_fresh[], double qoff_torso[]);
void calibration_adjustment_fsr(int fsr_today[], int vref_today[]);
void calibration_check_imu(double qoff_torso[]);
void filter_robotData(std::deque<RobotDataToRead> &deque_readIn, RobotDataToRead *readIn);
void robot_reconstruction(int num, mjModel *mm, mjData *dSim, mjData *dExp,
                          ContactDataAnalysis *cda_exp, ExpContactDataAnalysis *expCDA,
                          RobotDataToRead *read_in);

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

// Data and functions for optimisation based controller
unsigned int stateSIG_ID = mjSTATE_PHYSICS;
mjModel *m_ID = nullptr;                                            // model used for model based control
mjData *d_ID = nullptr;                                             // data used for model based control
int sOfst = 48;                                                     // Sensor number offset. See bioloid_5 XML
int cfs_idx[6] = {48 + 1, 48 + 2, 48 + 3, 48 + 8, 48 + 9, 48 + 10}; // 1:4 and 8:11
int ncfs = 6;                                                       // Two feet fx fy fz
double W_tauB = 1.;
double W_cfs = 0.3;
double W_pc_acc = 0.2;

// Residuals vector for optimisation based controller
void myCostFunc(const double dv_ip[12], double res_op[18], double jac_res_op[18 * 12], const double Amat[6 * 12], const mjData *dd);

// Ceres cost function class
class CtrlCostFunc : public ceres::SizedCostFunction<18, 12>
{
public:
    CtrlCostFunc(const double *Amat, const mjData *d) : d_(d), Amat_(Amat) {}

    bool Evaluate(double const *const *parameters,
                  double *residuals,
                  double **jacobians) const override
    {
        const double *dv_ip = parameters[0];

        myCostFunc(
            dv_ip,
            residuals,
            jacobians ? jacobians[0] : nullptr,
            Amat_,
            d_);
        return true;
    }

private:
    const double *Amat_;
    const mjData *d_;
};

void controller_SS_OC(const mjModel *m, mjData *d, double err_axisA[4], double err_w[3], double des_qj[12], double des_qjdot[12]);
void controller_pd(const mjModel *mm, mjData *dd, const double q_ctrl[], const double qdot_ctrl[],
                   const double KpGains[], const double KdGains[], const double sat_torques[],
                   const int qstartID, const int q_len);
void controller_IK_pd_motionexample1(const mjModel *mm, mjData *dd, double tt, double qjans_ip[], double dqjans_ip[]);
void controller_IK_pd_motionexample2(const mjModel *mm, mjData *dd, double tt, double qjans_ip[], double dqjans_ip[]);
void controller_IK_pd_motionexample4(const mjModel *mm, mjData *dd, double tt, double qjans_ip[], double dqjans_ip[]);
void controller_IK_pd_motionexample5(const mjModel *mm, mjData *dd, double tt, double qjans_ip[], double dqjans_ip[]);
void controller_IK_pd_motionexample6(const mjModel *mm, mjData *dd, double tt, double qjans_ip[], double dqjans_ip[]);
void controller_IK_pd_motionexample7(const mjModel *mm, mjData *dd, double tt);
void controller_wrapper();

// Data export variable
const int SIZE_MAX_DATA_SAMPLES = 2400;
double DataExport[SIZE_MAX_DATA_SAMPLES][200]; // 1000-2000 samples, 120 variables

// Colour definitions
float bgColor[4] = {0.2f, 0.2f, 0.2f, 0.8f}; // Dark grey with some transparency
float textColor[3] = {1.0f, 1.0f, 1.0f};     // White text
float myRED[4] = {1.0f, 0, 0, 0.8f};
float myGREEN[4] = {0, 1.0f, 0, 0.8f};
float myBLUE[4] = {0, 0, 1.0f, 0.8f};
float myMAGENTA[4] = {1.0f, 0, 1.0f, 0.8f};
float myCYAN[4] = {0, 1.0f, 1.0f, 0.8f};
float myYELLOW[4] = {1.0f, 1.0f, 0, 0.8f};

// Variables with dependencies outside the file
int vRef_today[9];           // 10-bit ADC Vref readings for both the feet and the force_stick.
int fsr_noLoad_today[9];     // 10-bit ADC preloaded sensor readings for both the feet and the force_stick.
double quat_offset_torso[4]; // Quaternion offset for the torso IMU

// MuJoCo data structures
mjModel *m = NULL;             // MuJoCo model
mjData *d_sim = NULL;          // MuJoCo data corresponding to the simulation
ContactDataAnalysis mycda_sim; // Contact data analysis structure
mjvCamera cam;                 // abstract camera
mjvOption vOpt;                // visualization options (common for both the scenes)
mjvScene scn_sim;              // abstract scene corresponding to the simulation
mjvScene scn_exp;              // abstract scene corresponding to the experiment
mjrContext con;                // custom GPU context

// Synchronisation variables: sim and keyboard callbacks in main thread
// std::mutex mtx_d_sim;

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

// Trial
std::atomic_uint8_t trial_num;

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

int filter_flag = 1;
int recons_num = 0;             // Reconstruction method number
int RECONS_Nmodes = 2;          // Number of reconstruction methods
double feedback_delay = 0.0047; // 4.7 ms

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
bool record_data_flag = false; // Data recording

// Only simulation thread reads and writes these variables.
double qjans_hold[12];  // qjans_hold is read out from file using load_qpos
double dqjans_hold[12]; // Set zero at start
double qjans[12];       // Used by IK function
double dqjans[12];      // Used by IK function

// For filtering (used by communication thread only.
double old_q_filtered[12], old_qdot_filtered[12];

// For IK, CoM initialized through load_sitepose() and load_com_pos function
double txyz[3], tkphi[4], lfxyz[3], lfkphi[4], rfxyz[3], rfkphi[4], com_pos[3];

// For joystick control
int jsSiteIdx = 0;                       // Site index for joystick control: 0-Torso, 1-LF, 2-RF
int jsMode = 0;                          // Joystick mode: 0-XYZ translation, 1-RPY rotation
bool button4state = 0, button5state = 0; // Button states
int jsMaxIncrXYZ[2] = {-100, 100};
int jsMaxIncrRPY[2] = {-1000, 1000};
std::atomic<int> jsTx, jsTy, jsTz, jsTpitch, jsTroll, jsTyaw; // Torso increments
std::atomic<int> jsLx, jsLy, jsLz, jsLpitch, jsLroll, jsLyaw; // Left foot increments
std::atomic<int> jsRx, jsRy, jsRz, jsRpitch, jsRroll, jsRyaw; // Right foot increments

// For keyboard control
std::atomic<bool> kr_ss;      // Kinematic reconstruction or shadow simulation
std::atomic<bool> imu_fusion; // IMU fusion  ON/OFF

// PD Gains controlling the 12 DOFs of bioloid.
const double Kp = 6;
const double KpHip = 6;
const double Kd = 0.05;
// const double Kp_hold[12] = {Kp, Kp, Kp * 1.2, Kp * 1.2, Kp, Kp, Kp, Kp, Kp * 1.2, Kp * 1.2, Kp, Kp};
const double Kp_hold[12] = {Kp, Kp, Kp, Kp, Kp, Kp, Kp, Kp, Kp, Kp, Kp, Kp};

const double Kd_hold[12] = {Kd, Kd, Kd, Kd, Kd / 5, Kd / 5, Kd, Kd, Kd, Kd, Kd / 5, Kd / 5};
const double Kd2 = 0.08;
const double Kd_hold_2[12] = {Kd2, Kd2, Kd2, Kd2, Kd2 / 5, Kd2 / 5, Kd2, Kd2, Kd2, Kd2, Kd2 / 5, Kd2 / 5};

// Saturation torques
const double Tsat = 1.3; // 1.3Nm stall torque, PITCH: 0.292 or 0.146 ROLL: 0.167 or 0.083 at ankle SSP/DSP
const double Tm = 5;
const double T_hold[12] = {Tsat, Tsat, Tsat, Tsat, Tsat, Tsat, Tsat, Tsat, Tsat, Tsat, Tsat, Tsat};
const double Tmax_hold[12] = {Tm, Tm, Tm, Tm, Tsat, Tsat, Tm, Tm, Tm, Tm, Tsat, Tsat};
const double Tmax_1[12] = {Tm, Tm, Tm, Tm, Tm, Tm, Tm, Tm, Tm, Tm, Tm, Tm};

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

    // Check for a connected joystick
    if (!glfwJoystickPresent(GLFW_JOYSTICK_1))
    {
        std::cout << "No joystick connected" << std::endl;
    }

    // Set all the atomic variables corresponding to joystick to 0
    jsTx.store(0.);
    jsTy.store(0.);
    jsTz.store(0.);
    jsTpitch.store(0.);
    jsTroll.store(0.);
    jsTyaw.store(0.);

    jsLx.store(0.);
    jsLy.store(0.);
    jsLz.store(0.);
    jsLpitch.store(0.);
    jsLroll.store(0.);
    jsLyaw.store(0.);

    jsRx.store(0.);
    jsRy.store(0.);
    jsRz.store(0.);
    jsRpitch.store(0.);
    jsRroll.store(0.);
    jsRyaw.store(0.);
}

void update_joystick_data()
{
    // Get the joystick data
    int axesCount, buttonCount;
    const float *axes = glfwGetJoystickAxes(GLFW_JOYSTICK_1, &axesCount);
    const unsigned char *buttons = glfwGetJoystickButtons(GLFW_JOYSTICK_1, &buttonCount);

    // Check if the joystick is connected
    if (axes == NULL || buttons == NULL)
    {
        std::cout << "Joystick disconnected" << std::endl;
        return;
    }

    // Print the axes values
    // for (int i = 0; i < axesCount; i++)
    // {
    //     printf("Axis %d: %f\t", i, axes[i]);
    // }
    // printf("\n");

    // Toggle between sites each time button #4 is pressed
    int siteList[3] = {TORSO_SITE, LF_SITE, RF_SITE};
    if (buttons[4])
    {
        if (!button4state)
        {
            jsSiteIdx = (jsSiteIdx + 1) % 3;
            printf("Site: %d\n", siteList[jsSiteIdx]);
        }
        button4state = true;
    }
    else
    {
        button4state = false;
    }

    // Toggle between XYZ translation mode and RPY rotation mode
    if (buttons[5])
    {
        if (!button5state)
        {
            jsMode ^= 1;
            printf("Mode: %d\n", jsMode);
        }
        button5state = true;
    }
    else
    {
        button5state = false;
    }

    // Based on the site update the atomic variables
    int temp;
    switch (siteList[jsSiteIdx])
    {
    case TORSO_SITE:
        if (!jsMode)
        {
            // Define increment in X Y Z axes based on the joystick axes
            temp = mapValue(axes[0], -1, 1, jsMaxIncrXYZ[0], jsMaxIncrXYZ[1]); // X incr
            jsTx.store(temp);                                                  // X incr
            temp = mapValue(axes[1], -1, 1, jsMaxIncrXYZ[0], jsMaxIncrXYZ[1]); // Y incr
            jsTy.store(temp);                                                  // Y incr
            temp = mapValue(axes[4], -1, 1, jsMaxIncrXYZ[0], jsMaxIncrXYZ[1]); // Z incr
            jsTz.store(temp);                                                  // Z incr
        }
        else
        {
            // Define the rotation based on the joystick axes
            temp = mapValue(axes[0], -1, 1, jsMaxIncrRPY[0], jsMaxIncrRPY[1]); // Roll incr
            jsTroll.store(temp);                                               // Roll incr
            temp = mapValue(axes[1], -1, 1, jsMaxIncrRPY[0], jsMaxIncrRPY[1]); // Pitch incr
            jsTpitch.store(temp);                                              // Pitch incr
            temp = mapValue(axes[4], -1, 1, jsMaxIncrRPY[0], jsMaxIncrRPY[1]); // Yaw incr
            jsTyaw.store(temp);                                                // Yaw incr
        }
        break;

    case LF_SITE:
        if (!jsMode)
        {
            // Define increment in X Y Z axes based on the joystick axes
            temp = mapValue(axes[0], -1, 1, jsMaxIncrXYZ[0], jsMaxIncrXYZ[1]); // X incr
            jsLx.store(temp);                                                  // X incr
            temp = mapValue(axes[1], -1, 1, jsMaxIncrXYZ[0], jsMaxIncrXYZ[1]); // Y incr
            jsLy.store(temp);                                                  // Y incr
            temp = mapValue(axes[4], -1, 1, jsMaxIncrXYZ[0], jsMaxIncrXYZ[1]); // Z incr
            jsLz.store(temp);                                                  // Z incr
        }
        else
        {
            // Define the rotation based on the joystick axes
            temp = mapValue(axes[0], -1, 1, jsMaxIncrRPY[0], jsMaxIncrRPY[1]); // Roll incr
            jsLroll.store(temp);                                               // Roll incr
            temp = mapValue(axes[1], -1, 1, jsMaxIncrRPY[0], jsMaxIncrRPY[1]); // Pitch incr
            jsLpitch.store(temp);                                              // Pitch incr
            temp = mapValue(axes[4], -1, 1, jsMaxIncrRPY[0], jsMaxIncrRPY[1]); // Yaw incr
            jsLyaw.store(temp);                                                // Yaw incr
        }
        break;

    case RF_SITE:
        if (!jsMode)
        {
            // Define increment in X Y Z axes based on the joystick axes
            temp = mapValue(axes[0], -1, 1, jsMaxIncrXYZ[0], jsMaxIncrXYZ[1]); // X incr
            jsRx.store(temp);                                                  // X incr
            temp = mapValue(axes[1], -1, 1, jsMaxIncrXYZ[0], jsMaxIncrXYZ[1]); // Y incr
            jsRy.store(temp);                                                  // Y incr
            temp = mapValue(axes[4], -1, 1, jsMaxIncrXYZ[0], jsMaxIncrXYZ[1]); // Z incr
            jsRz.store(temp);                                                  // Z incr
        }
        else
        {
            // Define the rotation based on the joystick axes
            temp = mapValue(axes[0], -1, 1, jsMaxIncrRPY[0], jsMaxIncrRPY[1]); // Roll incr
            jsRroll.store(temp);                                               // Roll incr
            temp = mapValue(axes[1], -1, 1, jsMaxIncrRPY[0], jsMaxIncrRPY[1]); // Pitch incr
            jsRpitch.store(temp);                                              // Pitch incr
            temp = mapValue(axes[4], -1, 1, jsMaxIncrRPY[0], jsMaxIncrRPY[1]); // Yaw incr
            jsRyaw.store(temp);                                                // Yaw incr
        }
        break;
    }

    // Print jsTx, jsTy, jsTz
    // printf("jsTx: %d\tjsTy: %d\tjsTz: %d\n", jsTx.load(), jsTy.load(), jsTz.load());
}

const char *filename = nullptr;
int main(int argc, const char **argv)
{
    // check command-line arguments
    if (argc != 2)
    {
        std::printf(" USAGE:  basic modelfile\n");
        return 0;
    }

    if (argc > 1)
    {
        filename = argv[1];
    }

    // load and compile model
    char error[1000] = "Could not load binary model";
    if (std::strlen(argv[1]) > 4 && !std::strcmp(argv[1] + std::strlen(argv[1]) - 4, ".mjb"))
    {
        m = mj_loadModel(filename, 0);
    }
    else
    {
        m = mj_loadXML(filename, 0, error, 1000);
    }
    if (!m)
    {
        mju_error("Load model error: %s", error);
    }
    // No need
    // m = mj_loadXML("../../model/bioloid_5/flat_ground_scene.xml", NULL, error, 1000);
    // m = mj_loadXML("../../model/bioloid_5/step_scene_left_up.xml", NULL, error, 1000);
    // m = mj_loadXML("../../model/bioloid_5/step_scene_right_up.xml", NULL, error, 1000);
    // m = mj_loadXML("../../model/bioloid_5/slopes_scene.xml", NULL, error, 1000);

    // d_sim is Global and modified only by the simulation thread
    d_sim = mj_makeData(m);

    // The robot states are shared memory locations.
    unsigned int my_spec = mjSTATE_TIME | mjSTATE_QPOS | mjSTATE_QVEL | mjSTATE_CTRL | mjSTATE_USERDATA;
    int nStateVar = mj_stateSize(m, my_spec);
    printf("State size: %d\n", nStateVar);

    // Initialise m_ID and d_ID for contact implicit inverse dynamics (SS + OC) controller
    mjSpec *specID = mj_parseXML(filename, nullptr, nullptr, 0);
    // Change the integrator to Euler for optimisation based controller
    if (specID)
    {
        specID->option.integrator = mjINT_EULER;
        specID->option.impratio = 1.;
        m_ID = mj_compile(specID, nullptr);
    }
    else
    {
        printf("Error parsing XML to mjSPEC for ID controller.\n");
        return -1;
    }
    d_ID = mj_makeData(m_ID);

    if (!d_ID)
    {
        printf("Error creating mjData for ID controller.\n");
        return -1;
    }

    // Allocate mjtNum array of nStateVar size
    mjtNum *state_sim_for_rendering = (mjtNum *)mju_malloc(nStateVar * sizeof(mjtNum));
    mjtNum *state_sim_for_comm = (mjtNum *)mju_malloc(nStateVar * sizeof(mjtNum));
    mjtNum *state_exp = (mjtNum *)mju_malloc(nStateVar * sizeof(mjtNum));

    // A copy of model for the rendering thread.
    mjModel *m_ren = mj_copyModel(NULL, m);

    // Copy of mjData for the rendering thread.
    mjData *d_sim_for_ren = mj_makeData(m);
    mjData *d_exp_for_ren = mj_makeData(m);
    ContactDataAnalysis mycda_sim_for_ren;
    ContactDataAnalysis mycda_exp_for_ren;
    RobotDataToRead readIn_for_ren;
    RobotDataToSend writeOut_for_ren;

    // Homing the force sensors
    communication_initialisation(fsr_noLoad_today, vRef_today, quat_offset_torso);

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
    kr_ss.store(true);       // Start the rendering thread
    imu_fusion.store(false); // IMU fusion is turned off by default

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
        copyCDAFromDoubleArray(&mycda_sim_for_ren, d_sim_for_ren->userdata + ADDR_CDA); // check later if updateContactDataAnalysis() is faster

        mjv_updateScene(m_ren, d_sim_for_ren, &vOpt, NULL, &cam, mjCAT_ALL, &scn_sim);
        // Foot COPs and GCOP box display
        drawCOMproj(m_ren, d_sim_for_ren, &scn_sim, &vOpt);
        drawPointBox(m_ren, d_sim_for_ren, mycda_sim_for_ren.lf_cop, 0.005, myBLUE, &scn_sim, &vOpt);
        drawPointBox(m_ren, d_sim_for_ren, mycda_sim_for_ren.rf_cop, 0.005, myBLUE, &scn_sim, &vOpt);
        drawPointBox(m_ren, d_sim_for_ren, mycda_sim_for_ren.gcop, 0.005, myMAGENTA, &scn_sim, &vOpt);

        mj_forward(m_ren, d_exp_for_ren);
        copyCDAFromDoubleArray(&mycda_exp_for_ren, d_exp_for_ren->userdata + ADDR_CDA);
        memcpy(&readIn_for_ren, d_exp_for_ren->userdata + ADDR_ROBOT_DATA_TO_READ, sizeof(RobotDataToRead));
        memcpy(&writeOut_for_ren, d_exp_for_ren->userdata + ADDR_ROBOT_DATA_TO_SEND, sizeof(RobotDataToSend));

        mjv_updateScene(m_ren, d_exp_for_ren, &vOpt, NULL, &cam, mjCAT_ALL, &scn_exp);
        drawCOMproj(m_ren, d_exp_for_ren, &scn_exp, &vOpt);
        // Foot COPs and GCOP box display: Reconstructed robot state
        drawPointBox(m_ren, d_exp_for_ren, mycda_exp_for_ren.lf_cop, 0.005, myBLUE, &scn_exp, &vOpt);
        drawPointBox(m_ren, d_exp_for_ren, mycda_exp_for_ren.rf_cop, 0.005, myBLUE, &scn_exp, &vOpt);
        drawPointBox(m_ren, d_exp_for_ren, mycda_exp_for_ren.gcop, 0.005, myMAGENTA, &scn_exp, &vOpt);
        // Foot COPs and GCOP box display: As measured by the force sensors. Relative foot location used from the reconstructed robot.
        double exp_lf_cop[3], exp_rf_cop[3], exp_gcop[3];
        exp_genCOP1(m_ren, d_exp_for_ren, &readIn_for_ren, 0, 0, exp_lf_cop, exp_rf_cop, exp_gcop);
        drawPointBox(m_ren, d_exp_for_ren, exp_lf_cop, 0.003, myBLUE, &scn_exp, &vOpt);
        drawPointBox(m_ren, d_exp_for_ren, exp_rf_cop, 0.003, myBLUE, &scn_exp, &vOpt);
        drawPointBox(m_ren, d_exp_for_ren, exp_gcop, 0.003, myMAGENTA, &scn_exp, &vOpt);
        // Show external force/torque on the robot
        drawPush(m_ren, d_exp_for_ren, &scn_exp, &vOpt, TORSO_BODY);

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
        // update_exp_COP_plot(&fig_gcz_trace_exp, m_ren, d_sim_for_ren, &mycda_sim_for_ren, &readIn_for_ren); // Feet position from sim + force from exp
        update_exp_COP_plot(&fig_gcz_trace_exp, m_ren, d_exp_for_ren, &mycda_exp_for_ren, &readIn_for_ren); // Feet position reconstructed + force from exp

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
        char text[200];
        // sim
        std::sprintf(text, "Simulation");
        mjrRect vp_title;
        // top of sim
        vp_title.left = vp_sim.left;
        vp_title.width = vp_sim.width;
        vp_title.height = vp_sim.height / 15;
        vp_title.bottom = vp_sim.bottom + vp_sim.height * 14 / 15;
        mjr_label(vp_title, mjFONT_NORMAL, text, bgColor[0], bgColor[1], bgColor[2], bgColor[3], textColor[0], textColor[1], textColor[2], &con);
        // exp
        std::sprintf(text, "Shadow of the physical robot");
        // top of exp
        vp_title.left = vp_exp.left;
        vp_title.width = vp_exp.width;
        vp_title.height = vp_exp.height / 15;
        vp_title.bottom = vp_exp.bottom + vp_exp.height * 14 / 15;
        mjr_label(vp_title, mjFONT_NORMAL, text, bgColor[0], bgColor[1], bgColor[2], bgColor[3], textColor[0], textColor[1], textColor[2], &con);
        // Pause/ Play text
        if (ppause)
        {
            std::sprintf(text, "PAUSED");
        }
        else
        {
            std::sprintf(text, "RUNNING");
        }
        // bottom left of sim
        mjrRect vp_pp;
        vp_pp.left = vp_sim.left /* + vp_sim.width * 3 / 4 */;
        vp_pp.width = vp_sim.width / 4;
        vp_pp.height = vp_sim.height / 15;
        vp_pp.bottom = vp_sim.bottom;
        mjr_label(vp_pp, mjFONT_NORMAL, text, bgColor[0], bgColor[1], bgColor[2], bgColor[3], textColor[0], textColor[1], textColor[2], &con);
        // Filter flag
        if (filter_flag)
        {
            std::sprintf(text, "Filtering ON");
        }
        else
        {
            std::sprintf(text, "Filtering OFF");
        }
        // top left of experiment section
        mjrRect vp_ff;
        vp_ff.left = vp_exp.left;
        vp_ff.width = vp_exp.width / 4;
        vp_ff.height = vp_exp.height / 15;
        vp_ff.bottom = vp_title.bottom - vp_exp.height / 15;
        mjr_label(vp_ff, mjFONT_NORMAL, text, bgColor[0], bgColor[1], bgColor[2], bgColor[3], textColor[0], textColor[1], textColor[2], &con);
        // IMU fusion ON/OFF
        if (1 == kr_ss.load())
        {
            std::sprintf(text, "Kinematic reconstruction");
        }
        else
        {
            if (1 == imu_fusion.load())
            {
                std::sprintf(text, "JA Shadow + IMU Fusion");
            }
            else
            {
                std::sprintf(text, "JA Shadow");
            }
        }
        // below filtering status
        mjrRect vp_imu;
        vp_imu.left = vp_exp.left /* + vp_exp.width * 3 / 4 */;
        vp_imu.width = vp_exp.width / 4;
        vp_imu.height = vp_exp.height / 15;
        vp_imu.bottom = vp_title.bottom - vp_exp.height * 2 / 15;
        mjr_label(vp_imu, mjFONT_NORMAL, text, bgColor[0], bgColor[1], bgColor[2], bgColor[3], textColor[0], textColor[1], textColor[2], &con);

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
        update_joystick_data();
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
    // mj_deleteData(d_exp);

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
    bool simulate_till_rendering_target = false;
    bool simulate_till_comm_target = false;
    unsigned int tmp;

    // Initialise the MuJoCo robot state from a file
    load_qpos(m, d_sim);
    load_sitepose(txyz, tkphi, lfxyz, lfkphi, rfxyz, rfkphi);
    // estimate for warmstarting IK solver
    mju_copy(qjans_hold, d_sim->qpos + 7, 12);
    mju_zero(dqjans_hold, 12);
    mju_copy(qjans, qjans_hold, 12);
    mju_copy(dqjans, dqjans_hold, 12);

    // Copy the state for communication for the first iteration of communication
    {
        // copyCDAToDoubleArray(d_sim->userdata + ADDR_CDA, &mycda_sim);
        std::lock_guard<std::mutex> lock(mtx_state_sim_for_comm);
        mj_getState(m, d_sim, StateSimSendToRobot, my_spec);
    }
    simulate_till_comm_target = false;
    cv_comm.notify_all();

    // Copy state for rendering for the first iteration of rendering
    {
        // copyCDAToDoubleArray(d_sim->userdata + ADDR_CDA, &mycda_sim);
        std::lock_guard<std::mutex> lock(mtx_state_sim_for_rendering);
        mj_getState(m, d_sim, StateSimForRendering, my_spec);
    }
    simulate_till_rendering_target = false;
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
        {
            tmp = time_rendering.load(std::memory_order_acquire);
            target_rendering = static_cast<double>(tmp) / 10000 + 0.016;
            simulate_till_rendering_target = true;
        }

        // Update comm target time?
        if (!comm_over.test_and_set(std::memory_order_acquire))
        {
            tmp = time_comm.load(std::memory_order_acquire);
            target_comm = static_cast<double>(tmp) / 10000 + 0.02;
            simulate_till_comm_target = true;
        }

        // If no one needs the simulation to run, sleep for 500 us
        if (!(simulate_till_rendering_target || simulate_till_comm_target))
        {
            // Sleep for 500 us
            std::this_thread::sleep_for(std::chrono::microseconds(500));
        }

        // If both rendering and comms need simulation, rearrange the targets in ascending order
        if (simulate_till_rendering_target && simulate_till_comm_target)
        {
            if (target_rendering < target_comm) // rendering target is closer
            {
                // printf("r c\n");
                // Advance the simulation till the target rendering time
                while (d_sim->time < target_rendering)
                {

                    mj_step(m, d_sim);
                    updateContactDataAnalysis(m, d_sim, &mycda_sim);
                    controller_wrapper();
                }
                // Copy the simulated robot's state for rendering
                {
                    mju_copy(d_sim->userdata + ADDR_ROBOT_DESIRED_Q, qjans, 12);
                    mju_copy(d_sim->userdata + ADDR_ROBOT_DESIRED_QDOT + 12, dqjans, 12);
                    copyCDAToDoubleArray(d_sim->userdata + ADDR_CDA, &mycda_sim);
                    std::lock_guard<std::mutex> lock(mtx_state_sim_for_rendering);
                    mj_getState(m, d_sim, StateSimForRendering, my_spec);
                }
                simulate_till_rendering_target = false;
                cv_rendering.notify_all();

                // Advance the simulation till the target communication time
                while (d_sim->time < target_comm)
                {
                    mj_step(m, d_sim);
                    updateContactDataAnalysis(m, d_sim, &mycda_sim);
                    controller_wrapper();
                }
                // Copy the state for communication
                {
                    mju_copy(d_sim->userdata + ADDR_ROBOT_DESIRED_Q, qjans, 12);
                    mju_copy(d_sim->userdata + ADDR_ROBOT_DESIRED_QDOT + 12, dqjans, 12);
                    copyCDAToDoubleArray(d_sim->userdata + ADDR_CDA, &mycda_sim);
                    std::lock_guard<std::mutex> lock(mtx_state_sim_for_comm);
                    mj_getState(m, d_sim, StateSimSendToRobot, my_spec);
                }
                simulate_till_comm_target = false;
                cv_comm.notify_all();
            }
            else
            {
                // printf("c r\n");
                // Advance the simulation till the target communication time
                while (d_sim->time < target_comm)
                {
                    mj_step(m, d_sim);
                    updateContactDataAnalysis(m, d_sim, &mycda_sim);
                    controller_wrapper();
                }
                // Copy the state for communication
                {
                    mju_copy(d_sim->userdata + ADDR_ROBOT_DESIRED_Q, qjans, 12);
                    mju_copy(d_sim->userdata + ADDR_ROBOT_DESIRED_QDOT + 12, dqjans, 12);
                    copyCDAToDoubleArray(d_sim->userdata + ADDR_CDA, &mycda_sim);
                    std::lock_guard<std::mutex> lock(mtx_state_sim_for_comm);
                    mj_getState(m, d_sim, StateSimSendToRobot, my_spec);
                }
                simulate_till_comm_target = false;
                cv_comm.notify_all();

                // Advance the simulation till the target rendering time
                while (d_sim->time < target_rendering)
                {
                    mj_step(m, d_sim);
                    updateContactDataAnalysis(m, d_sim, &mycda_sim);
                    controller_wrapper();
                }
                // Copy the simulated robot's state for rendering
                {
                    mju_copy(d_sim->userdata + ADDR_ROBOT_DESIRED_Q, qjans, 12);
                    mju_copy(d_sim->userdata + ADDR_ROBOT_DESIRED_QDOT + 12, dqjans, 12);
                    copyCDAToDoubleArray(d_sim->userdata + ADDR_CDA, &mycda_sim);
                    std::lock_guard<std::mutex> lock(mtx_state_sim_for_rendering);
                    mj_getState(m, d_sim, StateSimForRendering, my_spec);
                }
                simulate_till_rendering_target = false;
                cv_rendering.notify_all();
            }
        }

        // If only rendering needs simulation
        if (simulate_till_rendering_target)
        {
            // printf("r\n");
            // measure time require for sim
            // auto start = std::chrono::high_resolution_clock::now();
            // Advance the simulation till the target rendering time
            // mtx_d_sim.lock();
            while (d_sim->time < target_rendering)
            {
                mj_step(m, d_sim);
                updateContactDataAnalysis(m, d_sim, &mycda_sim);
                controller_wrapper();
            }
            // mtx_d_sim.unlock();
            // measure time require for sim
            // auto end = std::chrono::high_resolution_clock::now();
            // int64_t td = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
            // printf("sim time: %ld\n", td);
            // Copy the state for rendering
            {
                mju_copy(d_sim->userdata + ADDR_ROBOT_DESIRED_Q, qjans, 12);
                mju_copy(d_sim->userdata + ADDR_ROBOT_DESIRED_QDOT + 12, dqjans, 12);
                copyCDAToDoubleArray(d_sim->userdata + ADDR_CDA, &mycda_sim);
                // mju_printMat(mycda_sim.lf_cop, 1, 3);
                // mju_printMat(mycda_sim.rf_cop, 1, 3);
                std::lock_guard<std::mutex> lock(mtx_state_sim_for_rendering);
                mj_getState(m, d_sim, StateSimForRendering, my_spec);
            }
            simulate_till_rendering_target = false;
            cv_rendering.notify_all();
        }

        // If only communication needs simulation
        if (simulate_till_comm_target)
        {
            // printf("c\n");
            // measure time require for sim
            // auto start = std::chrono::high_resolution_clock::now();
            // Advance the simulation till the target communication time
            // mtx_d_sim.lock();
            while (d_sim->time < target_comm)
            {
                mj_step(m, d_sim);
                updateContactDataAnalysis(m, d_sim, &mycda_sim);
                controller_wrapper();
            }
            // mtx_d_sim.unlock();
            // auto end = std::chrono::high_resolution_clock::now();
            // int64_t td = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
            // printf("c sim time: %ld\n", td);
            // Copy the state for communication
            {
                mju_copy(d_sim->userdata + ADDR_ROBOT_DESIRED_Q, qjans, 12);
                mju_copy(d_sim->userdata + ADDR_ROBOT_DESIRED_QDOT + 12, dqjans, 12);
                copyCDAToDoubleArray(d_sim->userdata + ADDR_CDA, &mycda_sim);
                std::lock_guard<std::mutex> lock(mtx_state_sim_for_comm);
                mj_getState(m, d_sim, StateSimSendToRobot, my_spec);
            }
            simulate_till_comm_target = false;
            cv_comm.notify_all();
        }
    }
}

/**************************************************************************************/
// Initialise the communication to the robot. Force stick support suspended at the moment.
void communication_initialisation(int fsr_fresh[], int vref_fresh[], double qoff_torso[])
{
    int calibration_initialisation = 0;
    while (1)
    {
        printf("Skip calibration? Press 'y' to skip, presss Enter to continue.\n");
        calibration_initialisation = getchar();
        if (calibration_initialisation == 121) // ASCII for 'y'
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
            calibration_adjustment_fsr(fsr_fresh, vref_fresh);

            // communication_initialisation_forcestick(fsr_fresh, vref_fresh);

            printf("vRef_today: %d %d %d %d %d %d %d %d %d\n", vref_fresh[0],
                   vref_fresh[1], vref_fresh[2], vref_fresh[3], vref_fresh[4],
                   vref_fresh[5], vref_fresh[6], vref_fresh[7], vref_fresh[8]);

            calibration_check_imu(qoff_torso);

            printf("\nSatisfied with the calibration? Press Enter to continue. Press any other key to redo the calibration.\n");
            calibration_initialisation = getchar();
            if (calibration_initialisation == 10) // ASCII for Enter
                break;
        }
    }
}

// This function adjusts the force sensor preload that is used to compute the normal ground reaction force readings.
void calibration_adjustment_fsr(int fsr_today[], int vref_today[])
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
        initIO.txObj(packet_send, 0, size_packet_send); // Load the packet in the Tx buffer
        initIO.sendData(size_packet_send);              // Send the packet.
        usleep(DATA_OUT_SLEEP);                         // Sleep till the data is sent out.

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

// This function checks if the IMU has booted and the calibration has been loaded
void calibration_check_imu(double qoff_torso[])
{
    // Initiate the communication to the robot
    SerialTransfer::SerialTransfer initIO("/dev/ttyUSB0", 1000000);

    // Define call packets to be sent to the microcontrollers on the robot: {From,
    // To, Function Code}
    uint8_t packet_send[3], size_packet_send = 3;

    // Define the packet to receive data from the robot.
    uint8_t packet_received[64], size_packet_received = 0;
    int cSys, cGyr, cAcc, cMag;
    double quat_temp[4];
    const double quat_scale = (1.0 / (1 << 14)); // Unitless

    // Build the packet to send to the torso.
    packet_send[0] = PC_add;           // From
    packet_send[1] = T_add;            // To
    packet_send[2] = Req_imu_cal_data; // Function code

    printf("Make sure the robot torso is horizontal and press Enter.");
    getchar();
    for (int i = 0; i < 10; i++)
    {
        // Flush the serial port.
        initIO.reset();
        // Broadcast the packet.
        printf("Sending packet to the torso.\n");
        initIO.txObj(packet_send, 0, size_packet_send); // Load the packet in the Tx buffer
        initIO.sendData(size_packet_send);              // Send the packet.
        usleep(DATA_OUT_SLEEP);                         // Sleep till the data is sent out.

        // Receive the packet from the torso.
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

                // Read data, forget from:0, to:1, FC:2 bytes
                cSys = packet_received[3];
                cGyr = packet_received[4];
                cAcc = packet_received[5];
                cMag = packet_received[6];
                int16_t u, x, y, z;
                u = ((uint16_t)packet_received[7]) | (((uint16_t)packet_received[8]) << 8);
                x = ((uint16_t)packet_received[9]) | (((uint16_t)packet_received[10]) << 8);
                y = ((uint16_t)packet_received[11]) | (((uint16_t)packet_received[12]) << 8);
                z = ((uint16_t)packet_received[13]) | (((uint16_t)packet_received[14]) << 8);
                quat_temp[0] = ((double)u) * quat_scale;
                quat_temp[1] = ((double)x) * quat_scale;
                quat_temp[2] = ((double)y) * quat_scale;
                quat_temp[3] = ((double)z) * quat_scale;
                // Print on screen
                printf("IMU cal: S:%d G:%d A:%d M:%d\t", cSys, cGyr, cAcc, cMag);
                printf("Quat: %lf %lf %lf %lf\n", quat_temp[0], quat_temp[1], quat_temp[2], quat_temp[3]);
            }

            // Check for timeout.
            auto time_difference = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - time_start).count();
            if (time_difference > RS485_TO)
            {
                printf("Timeout. No packet received from the left foot.\n");
                break;
            }
        }
    }

    printf("Press Enter to record the IMU calibration data.\n");
    getchar();

    mju_copy4(qoff_torso, quat_temp);
}
/**************************************************************************************/

double err_Angle=0;

void communication_with_robot(mjtNum *StateSimSendToRobot, mjtNum *StateExp)
{
    // Open the COM port and initiate the communication.
    SerialTransfer::SerialTransfer myrobotIO("/dev/ttyUSB0", 1000000);

    // Data recording happens in this thread
    int nloops_rec = 0;
    int data_rec_width = 0;

    // Copy of mjModel and mjData_sim for the communication thread.
    mjModel *m_comm = mj_copyModel(NULL, m);
    unsigned int my_spec = mjSTATE_TIME | mjSTATE_QPOS | mjSTATE_QVEL | mjSTATE_CTRL | mjSTATE_USERDATA;
    mjData *d_sim_for_comm = mj_makeData(m_comm);
    ContactDataAnalysis mycda_sim_for_comm;

    // mjData corresponding to the reconstructed state of the robot based on sensor data.
    mjData *d_exp = mj_makeData(m_comm);
    ContactDataAnalysis mycda_exp;
    ExpContactDataAnalysis myexpcda;
    bool gcz_inside = true; // Flag to indicate if the COP-ZMP is beyond the safe zone.

    // Initialise the RobotoDataToSend structure.
    RobotDataToSend writeOut;
    // Zero initialise the RobotDataToRead structure.
    RobotDataToRead readIn_filtered;
    RobotDataToRead readIn_raw;

    // Define call packets to be sent to the microcontrollers on the robot: {FromTo, Function Code}
    uint8_t packet_send[51], size_packet_send = 51; // 3 + 48(q12x2|qdot12x2) = 51

    // Define the packet to receive data from the robot.
    uint8_t packet_received[64], size_packet_received = 64;
    int replies = 0;

    // Filter window for raw data.
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

        // printf("glfwtime: %lg\n",glfwGetTime());

        // Reset the loop timer
        loop_time_start = std::chrono::steady_clock::now();

        // Copy the simulation state to be sent to the robot.
        std::unique_lock<std::mutex> sim_comm_lck(mtx_state_sim_for_comm);
        cv_comm.wait_for(sim_comm_lck, std::chrono::milliseconds(1));
        // cv_comm.wait(sim_comm_lck);
        mj_setState(m_comm, d_sim_for_comm, StateSimSendToRobot, my_spec);
        sim_comm_lck.unlock();
        mj_forward(m_comm, d_sim_for_comm);

        copyCDAFromDoubleArray(&mycda_sim_for_comm, d_sim_for_comm->userdata + ADDR_CDA);
        // Conditional copy based on COP-ZMP criteria
        // if (check_interior(mycda_exp.gcop, 0, 0, 0.05, 0.04) && gcz_inside)
        // {
        //     mju_copy(writeOut.q, (d_sim_for_comm->qpos + 7), 12);
        //     mju_copy(writeOut.qdot, (d_sim_for_comm->qvel + 6), 12);
        // }
        // else
        // {
        //     mju_copy(writeOut.q, (d_sim_for_comm->qpos + 7), 12);
        //     mju_copy(writeOut.qdot, (d_sim_for_comm->qvel + 6), 12);
        //     if (!gcz_inside)
        //         printf("Precheck: COP-ZMP hit limits.\n");
        // }

        mju_copy(writeOut.q, (d_sim_for_comm->qpos + 7), 12);
        mju_copy(writeOut.qdot, (d_sim_for_comm->qvel + 6), 12);

        // printf("qdot: ");
        // mju_printMat(writeOut.qdot, 1, 12);

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
        int whoreplied[4] = {0, 0, 0, 0};
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
                // Store the address of the microcontroller that replied.
                whoreplied[replies - 1] = packet_received[0];

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
                printf("TO: %d\n", replies);
                // print who replied
                for (int i = 0; i < 4; i++)
                {
                    printf("%d ", whoreplied[i]);
                }
                printf("\n");
                break;
            }
        }
        // printf("TO time: %lg\n", glfwGetTime());

        // Robot reconstruction: set d_exp as close to the actual robot as possible.
        robot_reconstruction(recons_num, m_comm, d_sim_for_comm, d_exp, &mycda_exp, &myexpcda, &readIn_filtered);

        // Check for fall based on sensor feedback
        // if (check_interior(mycda_exp.gcop, 0, 0, 0.05, 0.04))
        // { // Nothing to do
        // }
        // else
        // {
        //     gcz_inside = false;
        //     printf("Feedback: COP-ZMP hit limits.\n");
        // }
        // Raise a flag so that next time the robot is homed. Remember to reset the flag every iteration

        // Copy the readIn data + WriteOut data to d_exp->userdata
        memcpy(d_exp->userdata + ADDR_ROBOT_DATA_TO_READ, &readIn_filtered, sizeof(RobotDataToRead));
        memcpy(d_exp->userdata + ADDR_ROBOT_DATA_TO_SEND, &writeOut, sizeof(RobotDataToSend));
        copyCDAToDoubleArray(d_exp->userdata + ADDR_CDA, &mycda_exp);
        // Update the StateExp
        mtx_state_exp.lock();
        mj_getState(m_comm, d_exp, StateExp, my_spec);
        mtx_state_exp.unlock();

        // Save data to DataExport
        if (record_data_flag)
        {
            if (nloops_rec == 0)
                printf("R %lg\n", glfwGetTime());
            if (nloops_rec > SIZE_MAX_DATA_SAMPLES)
            {
                printf("DataExport full. Exiting.\n");
                break;
            }

            int ptr = 0;
            // Sim time
            DataExport[nloops_rec][ptr] = d_sim_for_comm->time;
            ptr++;

            // // Exp time
            // DataExport[nloops_rec][ptr] = d_exp->time;
            // ptr++;

            // Joint angles: desired
            // mju_copy(DataExport[nloops_rec] + ptr, d_sim_for_comm->userdata + ADDR_ROBOT_DESIRED_Q, 12);
            // ptr += 12;

            // // Joint angles: sim
            // mju_copy(DataExport[nloops_rec] + ptr, d_sim_for_comm->qpos + 7, 12);
            // ptr += 12;

            // // Joint angles: from robot
            // mju_copy(DataExport[nloops_rec] + ptr, readIn_filtered.q, 12);
            // ptr += 12;

            // // Joint angles: shadow simulation
            // mju_copy(DataExport[nloops_rec] + ptr, d_exp->qpos + 7, 12);
            // ptr += 12;

            // // Torso site position: sim
            // mju_copy(DataExport[nloops_rec] + ptr, d_sim_for_comm->site_xpos + 3 * TORSO_SITE, 3);
            // ptr += 3;

            // // Torso site position: exp
            // mju_copy(DataExport[nloops_rec] + ptr, d_exp->site_xpos + 3 * TORSO_SITE, 3);
            // ptr += 3;

            // // Torso site orientation: sim
            // mju_copy(DataExport[nloops_rec] + ptr, d_sim_for_comm->site_xmat + 9 * TORSO_SITE, 9);
            // ptr += 9;

            // // Torso site orientation: exp
            // mju_copy(DataExport[nloops_rec] + ptr, d_exp->site_xmat + 9 * TORSO_SITE, 9);
            // ptr += 9;

            // // Torso orientation: exp IMU
            // mju_copy(DataExport[nloops_rec] + ptr, readIn_filtered.torso_quat, 4);
            // ptr += 4;

            // // Left foot site position: sim
            // mju_copy(DataExport[nloops_rec] + ptr, d_sim_for_comm->site_xpos + 3 * LF_SITE, 3);
            // ptr += 3;

            // // Left foot site position: exp
            // mju_copy(DataExport[nloops_rec] + ptr, d_exp->site_xpos + 3 * LF_SITE, 3);
            // ptr += 3;

            // // Left foot site orientation: sim
            // mju_copy(DataExport[nloops_rec] + ptr, d_sim_for_comm->site_xmat + 9 * LF_SITE, 9);
            // ptr += 9;

            // // Left foot site orientation: exp
            // mju_copy(DataExport[nloops_rec] + ptr, d_exp->site_xmat + 9 * LF_SITE, 9);
            // ptr += 9;

            // // Left foot site orientaion: exp IMU
            // mju_copy(DataExport[nloops_rec] + ptr, readIn_filtered.lf_quat, 4);
            // ptr += 4;

            // // Right foot site position: sim
            // mju_copy(DataExport[nloops_rec] + ptr, d_sim_for_comm->site_xpos + 3 * RF_SITE, 3);
            // ptr += 3;

            // // Right foot site position: exp
            // mju_copy(DataExport[nloops_rec] + ptr, d_exp->site_xpos + 3 * RF_SITE, 3);
            // ptr += 3;

            // // Right foot site orientation: sim
            // mju_copy(DataExport[nloops_rec] + ptr, d_sim_for_comm->site_xmat + 9 * RF_SITE, 9);
            // ptr += 9;

            // // Right foot site orientation: exp
            // mju_copy(DataExport[nloops_rec] + ptr, d_exp->site_xmat + 9 * RF_SITE, 9);
            // ptr += 9;

            // // Right foot site orientation: exp IMU
            // mju_copy(DataExport[nloops_rec] + ptr, readIn_filtered.rf_quat, 4);
            // ptr += 4;

            // COP-ZMP: sim
            // printf("save: %.3f %.3f %.3f\n", mycda_sim_for_comm.gcop[0], mycda_sim_for_comm.gcop[1], mycda_sim_for_comm.gcop[2]);
            mju_copy(DataExport[nloops_rec] + ptr, mycda_sim_for_comm.gcop, 3);
            ptr += 3;

            // COP-ZMP: exp
            mju_copy(DataExport[nloops_rec] + ptr, mycda_exp.gcop, 3);
            ptr += 3;

            // COP-ZMP: exp sensors
            mju_copy(DataExport[nloops_rec] + ptr, myexpcda.gCOP, 3);
            ptr += 3;

            // Error in the orietation of the torso between the simulation and the robot.
            *(DataExport[nloops_rec] + ptr) = err_Angle;
            ptr += 1;

            // Increment the recorded loop counter
            nloops_rec++;
            data_rec_width = ptr;
        }

        // print the time since last loop
        // current_time = std::chrono::steady_clock::now();
        // double loop_time = std::chrono::duration_cast<std::chrono::microseconds>(current_time - last_time).count();
        // printf("Comm loop ms: %.3f\n", loop_time / 1000.0);
        // last_time = current_time;

        // Sleep till LOOP_TIME is over.
        current_time = std::chrono::steady_clock::now();
        while (std::chrono::duration_cast<std::chrono::microseconds>(current_time - loop_time_start).count() < int64_t(LOOP_TIME))
        {
            std::this_thread::sleep_for(std::chrono::microseconds(50));
            current_time = std::chrono::steady_clock::now();
        }

        // printf("end time: %lg\n", glfwGetTime());
        // Compute the time taken for the loop.
        // current_time = std::chrono::steady_clock::now();
        // auto loop_time = std::chrono::duration_cast<std::chrono::microseconds>(current_time - loop_time_start).count();
        // // Print the loop time
        // std::cout << "Loop time: " << loop_time << std::endl;
    }

    FILE *fp;
    fp = fopen("cps_data.txt", "w");
    if (fp == NULL)
    {
        printf("Error opening file!\n");
        exit(1);
    }
    else
    {
        printf("File opened successfully. Saving recorded data to the file.\n");
    }

    printf("nloops_rec: %d, data_rec_width: %d\n", nloops_rec, data_rec_width);

    // Write data recorded in DataExport to a file.
    for (int i = 0; i < nloops_rec; i++)
    {
        for (int j = 0; j < data_rec_width; j++)
        {
            fprintf(fp, "%.4f\t", DataExport[i][j]);
        }
        fprintf(fp, "\n");
    }
    // Close the file used for saving data.
    int fileclosed = fclose(fp);
    printf("Closed the file. %d\n", fileclosed);

    // Free resources
    mj_deleteData(d_sim_for_comm);
    mj_deleteData(d_exp);
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

int onlyonce = 1;
void robot_reconstruction(int num, mjModel *mm, mjData *dSim, mjData *dExp,
                          ContactDataAnalysis *cda_exp, ExpContactDataAnalysis *expCDA,
                          RobotDataToRead *read_in)
{
    // Perform kinematic reconstruction at the start of the simulation regardless of 'kr_ss' value.
    if (onlyonce)
    {
        onlyonce = 0;
        // Update the reconstructed robot position based on read_in data + reference dSim data.
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

            // Set the q_old_filtered value
            mju_copy(old_q_filtered, dExp->qpos + 7, 12);
            // Handle timing
            dExp->time = dSim->time + feedback_delay;
        }
    }

    // Perform state according to the choice of 'kr_ss' variable.

    // If kinematic reconstruction is selected,
    if (1 == kr_ss.load())
    {
        // Set this to 1 so that next time also kinematic reconstruction is performed.
        onlyonce = 1;
    }
    else
    {
        // Shadow simulation

        // Handle timing
        dExp->time = dSim->time + feedback_delay;
        double tstart = dExp->time;

        // Extrapolate q and qdot. Define: t_old t_new, q_old, q_new, qdot_old, qdot_new
        double t_new = dExp->time;
        double t_old = t_new - 0.02; // okay even if it is negative

        double q_old[12], q_new[12], extrp_q[12];
        mju_copy(q_old, old_q_filtered, 12);
        mju_copy(q_new, read_in->q, 12);
        // q_new- q_old
        double q_new_m_old[12];
        mju_sub(q_new_m_old, q_new, q_old, 12);
        // (q_new - q_old)/(t_new - t_old)
        double my_qdot[12];
        mju_scl(my_qdot, q_new_m_old, 1 / (t_new - t_old), 12);
        // q_old - my_qdot*t_old
        double q_cons[12];
        mju_scl(q_cons, my_qdot, t_old, 12);
        mju_sub(q_cons, q_old, q_cons, 12);
        double zz[12] = {0};

        // while (dExp->time < (tstart + 0.02 - feedback_delay))
        // Loop till little ahead of real time to bridge the gap till the feedback of the next iteration
        while (dExp->time < (tstart + 0.02))
        {
            // Extrapolate q and qdot (qdot not used at the moment)
            mju_scl(extrp_q, my_qdot, dExp->time, 12);
            mju_add(extrp_q, extrp_q, q_cons, 12);
            // Compute the orientation error between IMU and shadow sim*****************************************************************

            // IMU torso orientation
            double imu_torso_quat[4];
            mju_copy4(imu_torso_quat, read_in->torso_quat);

            // Remove the initial offset (stored in quat_offset_torso[4])
            double imu_corrected_torso_quat[4], imu_corrected_torso_Rmat[9];
            my_mju_relQuat(imu_corrected_torso_quat, imu_torso_quat, quat_offset_torso);
            mju_quat2Mat(imu_corrected_torso_Rmat, imu_corrected_torso_quat);
            
            // Shadow robot dExp torso orientation
            double shadow_torso_Rmat[9];
            mju_quat2Mat(shadow_torso_Rmat, dExp->qpos + 3);
            
            // Error in the orientations
            double err_Rmat[9];
            mju_mulMatTMat(err_Rmat, shadow_torso_Rmat, imu_corrected_torso_Rmat, 3, 3, 3);
            double err_axis_angle[4];
            my_mju_mat2axisAngle(err_Rmat, err_axis_angle);
            // Saving to a global variable
            err_Angle = err_axis_angle[3];
            // printf("eA: %f\n", err_Angle);
            // fflush(stdout);

            // Compute the angular velocity error between the IMU and shadow sim*********************************************************
            double w_torso_imu[3], w_torso_shadow[3]; // both are expressed in the body fixed frame
            mju_copy3(w_torso_imu, read_in->torso_w);
            mju_copy3(w_torso_shadow, dExp->qvel + 3);

            double w_rel[3];
            // w_desired (sensed by IMU) - w_shadow
            mju_sub3(w_rel, w_torso_imu, w_torso_shadow);
            //***************************************************************************************************************************
            // SS + OC controller
            controller_SS_OC(mm, dExp, err_axis_angle, w_rel, extrp_q, zz);

            //***************************************************************************************************************************

            mj_step(mm, dExp);
        }

        // Update the old values to be used for extrapolation in the next iteration
        mju_copy(old_q_filtered, read_in->q, 12);
        mju_copy(old_qdot_filtered, read_in->qdot, 12);
    }

    // Updates CDA purely based on MuJoCo dExp
    updateContactDataAnalysis(mm, dExp, cda_exp);

    // Update GRFs and gCOP computed using the sensors
    updateExpContactDataAnalysis(mm, dExp, read_in, expCDA);
}

/**************************************************************************************/
void controller_pd(const mjModel *mm, mjData *dd, const double q_ctrl[],
                   const double qdot_ctrl[], const double KpGains[],
                   const double KdGains[], const double sat_torques[],
                   const int qstartID, const int q_len)
{
    double err, derr, op_torque;
    for (int i = 0; i < q_len; i++)
    {
        // Position error in the joint space = desired - current
        err = q_ctrl[i] - dd->qpos[qstartID + i];
        // Velocity error in the joint space (Assumption: the model has floating
        // base.)
        derr = qdot_ctrl[i] - dd->qvel[qstartID - 1 + i];
        // Set motor torque to remove the error
        op_torque = KpGains[i] * (err) + KdGains[i] * (derr);
        // Torque clipping
        if (abs(op_torque) > sat_torques[i])
            op_torque = sat_torques[i] * mju_sign(op_torque);
        // Write to control data structure
        dd->ctrl[i] = op_torque;

        // if (i == 5 || i == 11)
        //     printf("ankle err:%lg derr:%lg\n", err, derr);
    }
}

void controller_IK_pd_motionexample1(const mjModel *mm, mjData *dd, double tt, double qjans_ip[], double dqjans_ip[])
{
    double txyz_ip[3], lfxyz_ip[3], rfxyz_ip[3];
    double tkphi_ip[4] = {1, 0, 0, 0}, lfkphi_ip[4] = {1, 0, 0, 0}, rfkphi_ip[4] = {1, 0, 0, 0};
    double w = 2;
    txyz_ip[0] = 0 + txyz[0]; // torso
    txyz_ip[1] = 0 + txyz[1]; // torso
    txyz_ip[2] = 0 + txyz[2]; // torso

    // Sinusoidal pitching motion of the torso about x axis.
    // tkphi_ip[3] = (10 * M_PI / 180) * sin(w * tt); // torso

    mju_copy3(lfxyz_ip, lfxyz);   // left foot
    mju_copy4(lfkphi_ip, lfkphi); // left foot
    mju_copy3(rfxyz_ip, rfxyz);   // right foot
    mju_copy4(rfkphi_ip, rfkphi); // right foot

    // Inverse kinematics
    bioloid_12dof_IK_position_v2(mm, TORSO_SITE, LF_SITE, RF_SITE, txyz_ip, tkphi_ip, lfxyz_ip, lfkphi_ip, rfxyz_ip, rfkphi_ip, qjans_ip, 20, 1e-6, 1e-5);

    // Control 12 DOFs.
    controller_pd(mm, dd, qjans_ip, dqjans_ip, Kp_hold, Kd_hold, Tmax_1, 7, 12);
}

void controller_IK_pd_motionexample2(const mjModel *mm, mjData *dd, double tt, double qjans_ip[], double dqjans_ip[])
{
    double txyz_ip[3], lfxyz_ip[3], rfxyz_ip[3];
    double tkphi_ip[4], lfkphi_ip[4], rfkphi_ip[4];
    double Ke = 2 * M_PI, Te = 4, xtraj, ytraj, k;
    double A = 0.025 / (Ke);

    if (tt <= Te)
    {
        k = (-2 * Ke / pow(Te, 3)) * (pow(tt, 3)) + (3 * Ke / (pow(Te, 2))) * (pow(tt, 2));
        xtraj = A * k * cos(k);
        ytraj = A * k * sin(k);
        txyz_ip[0] = xtraj + txyz[0]; // torso
        txyz_ip[1] = ytraj + txyz[1]; // torso
        txyz_ip[2] = 0 + txyz[2];     // torso
    }
    else
    {
        tt = Te;
        k = (-2 * Ke / pow(Te, 3)) * (pow(tt, 3)) + (3 * Ke / (pow(Te, 2))) * (pow(tt, 2));
        xtraj = A * k * cos(k);
        ytraj = A * k * sin(k);
        txyz_ip[0] = xtraj + txyz[0]; // torso
        txyz_ip[1] = ytraj + txyz[1]; // torso
        txyz_ip[2] = 0 + txyz[2];     // torso
    }
    mju_copy4(tkphi_ip, tkphi);   // torso
    mju_copy3(lfxyz_ip, lfxyz);   // left foot
    mju_copy4(lfkphi_ip, lfkphi); // left foot
    mju_copy3(rfxyz_ip, rfxyz);   // right foot
    mju_copy4(rfkphi_ip, rfkphi); // right foot

    // Inverse kinematics
    bioloid_12dof_IK_position_v2(mm, TORSO_SITE, LF_SITE, RF_SITE, txyz_ip, tkphi_ip, lfxyz_ip, lfkphi_ip, rfxyz_ip, rfkphi_ip, qjans_ip, 20, 1e-6, 1e-5);

    // Control 12 DOFs.
    controller_pd(mm, dd, qjans_ip, dqjans_ip, Kp_hold, Kd_hold, Tmax_1, 7, 12);
}

void controller_IK_pd_motionexample3(const mjModel *mm, mjData *dd, double tt, double qjans_ip[], double dqjans_ip[])
{
    // Intentional fall: Go back 3 cm and rotate back by 10 degrees in 2 seconds. Hold the position after that.
    double txyz_ip[3], lfxyz_ip[3], rfxyz_ip[3];
    double tkphi_ip[4] = {1, 0, 0, 0}, lfkphi_ip[4] = {1, 0, 0, 0}, rfkphi_ip[4] = {1, 0, 0, 0};
    double Ke = (12 * M_PI / 180), Te = 2, ytraj, ztraj, k;
    double A = txyz[2];

    if (tt <= Te)
    {
        k = (-2 * Ke / pow(Te, 3)) * (pow(tt, 3)) + (3 * Ke / (pow(Te, 2))) * (pow(tt, 2));
        ytraj = 1 * A * sin(k);
        ztraj = A * (1 - cos(k));
        txyz_ip[0] = txyz[0];         // torso x
        txyz_ip[1] = ytraj + txyz[1]; // torso y
        txyz_ip[2] = ztraj + txyz[2]; // torso z
        tkphi_ip[3] = +1 * -k;        // torso
    }
    else
    {
        tt = Te;
        k = (-2 * Ke / pow(Te, 3)) * (pow(tt, 3)) + (3 * Ke / (pow(Te, 2))) * (pow(tt, 2));
        ytraj = 1 * A * sin(k);
        ztraj = A * (1 - cos(k));
        txyz_ip[0] = txyz[0];         // torso
        txyz_ip[1] = ytraj + txyz[1]; // torso
        txyz_ip[2] = ztraj + txyz[2]; // torso
        tkphi_ip[3] = +1 * -k;        // torso
    }
    mju_copy3(lfxyz_ip, lfxyz);   // left foot
    mju_copy4(lfkphi_ip, lfkphi); // left foot
    mju_copy3(rfxyz_ip, rfxyz);   // right foot
    mju_copy4(rfkphi_ip, rfkphi); // right foot

    // Inverse kinematics
    bioloid_12dof_IK_position_v2(mm, TORSO_SITE, LF_SITE, RF_SITE, txyz_ip, tkphi_ip, lfxyz_ip, lfkphi_ip, rfxyz_ip, rfkphi_ip, qjans_ip, 20, 1e-6, 1e-5);

    // Control 12 DOFs.
    controller_pd(mm, dd, qjans_ip, dqjans_ip, Kp_hold, Kd_hold, Tmax_1, 7, 12);
}

void controller_IK_pd_motionexample4(const mjModel *mm, mjData *dd, double tt, double qjans_ip[], double dqjans_ip[])
{
    // Hip tracing horizontal/vertical ellipse
    double txyz_ip[3], lfxyz_ip[3], rfxyz_ip[3];
    double tkphi_ip[4] = {1, 0, 0, 0}, lfkphi_ip[4] = {1, 0, 0, 0}, rfkphi_ip[4] = {1, 0, 0, 0};
    double Ke = 2 * M_PI, Te = 8, xtraj = 0, ytraj = 0, ztraj = 0, k = 0;
    double A = 0.016, B = 0.013;

    if (tt > Te)
        tt = Te;

    k = (-2 * Ke / pow(Te, 3)) * (pow(tt, 3)) + (3 * Ke / (pow(Te, 2))) * (pow(tt, 2));
    xtraj = A * cos(k + M_PI / 2);
    ytraj = -B + B * sin(k + M_PI / 2);
    // ztraj = -B + B * sin(k + M_PI / 2);
    txyz_ip[0] = xtraj + txyz[0]; // torso
    txyz_ip[1] = ytraj + txyz[1]; // torso
    txyz_ip[2] = ztraj + txyz[2]; // torso

    mju_copy4(tkphi_ip, tkphi);   // torso
    mju_copy3(lfxyz_ip, lfxyz);   // left foot
    mju_copy4(lfkphi_ip, lfkphi); // left foot
    mju_copy3(rfxyz_ip, rfxyz);   // right foot
    mju_copy4(rfkphi_ip, rfkphi); // right foot

    // Inverse kinematics
    bioloid_12dof_IK_position_v2(mm, TORSO_SITE, LF_SITE, RF_SITE, txyz_ip, tkphi_ip, lfxyz_ip, lfkphi_ip, rfxyz_ip, rfkphi_ip, qjans_ip, 20, 1e-6, 1e-5);

    // Control 12 DOFs.
    controller_pd(mm, dd, qjans_ip, dqjans_ip, Kp_hold, Kd_hold, Tmax_1, 7, 12);
}

void controller_IK_pd_motionexample5(const mjModel *mm, mjData *dd, double tt, double qjans_ip[], double dqjans_ip[])
{
    // Side to side sway: chirp 0 to 0.7 Hz in 20 seconds.
    double txyz_ip[3], lfxyz_ip[3], rfxyz_ip[3];
    double tkphi_ip[4] = {1, 0, 0, 0}, lfkphi_ip[4] = {1, 0, 0, 0}, rfkphi_ip[4] = {1, 0, 0, 0};
    double f = 0, fmin = 0., fmax = 0.3;
    double T = 1;

    // Linearly increase w from w_min to w_max in T seconds, reduce back to w_min in T seconds, and hold on to w_min after that.
    if (tt < T)
    {
        f = fmin + (fmax - fmin) * tt / T;
    }
    else
    {
        f = fmin;
        f = fmax;
    }
    // printf("tt: %lg \t w: %lg\n", tt, w);

    // Sinusoidal side to side motion of the torso along x axis.
    txyz_ip[0] = 0.015 * sin(2 * M_PI * f * tt) + txyz[0]; // torso
    txyz_ip[1] = 0 + txyz[1];                              // torso
    txyz_ip[2] = 0 + txyz[2];                              // torso

    mju_copy4(tkphi_ip, tkphi);   // torso
    mju_copy3(lfxyz_ip, lfxyz);   // left foot
    mju_copy4(lfkphi_ip, lfkphi); // left foot
    mju_copy3(rfxyz_ip, rfxyz);   // right foot
    mju_copy4(rfkphi_ip, rfkphi); // right foot

    // Inverse kinematics
    bioloid_12dof_IK_position_v2(mm, TORSO_SITE, LF_SITE, RF_SITE, txyz_ip, tkphi_ip, lfxyz_ip, lfkphi_ip, rfxyz_ip, rfkphi_ip, qjans_ip, 20, 1e-6, 1e-5);

    // Control 12 DOFs.
    controller_pd(mm, dd, qjans_ip, dqjans_ip, Kp_hold, Kd_hold, Tmax_1, 7, 12);
}

void controller_IK_pd_motionexample6(const mjModel *mm, mjData *dd, double tt, double qjans_ip[], double dqjans_ip[])
{
    int temp;
    double deno = 2000000;

    // Joystick controller
    temp = jsTx.load();
    txyz[0] += float(temp) / deno; // torso x

    temp = jsTy.load();
    txyz[1] += float(temp) / deno; // torso y

    temp = jsTz.load();
    txyz[2] += float(temp) / deno; // torso z

    temp = jsLx.load();
    lfxyz[0] += float(temp) / deno; // left foot x

    temp = jsLy.load();
    lfxyz[1] += float(temp) / deno; // left foot y

    temp = jsLz.load();
    lfxyz[2] += float(temp) / deno; // left foot z

    temp = jsRx.load();
    rfxyz[0] += float(temp) / deno; // right foot x

    temp = jsRy.load();
    rfxyz[1] += float(temp) / deno; // right foot y

    temp = jsRz.load();
    rfxyz[2] += float(temp) / deno; // right foot z

    // Inverse kinematics
    bioloid_12dof_IK_position_v2(mm, TORSO_SITE, LF_SITE, RF_SITE, txyz, tkphi, lfxyz, lfkphi, rfxyz, rfkphi, qjans_ip, 20, 1e-6, 1e-5);

    // Control 12 DOFs.
    controller_pd(mm, dd, qjans_ip, dqjans_ip, Kp_hold, Kd_hold, Tmax_1, 7, 12);
}

void controller_IK_pd_motionexample7(const mjModel *mm, mjData *dd, double tt)
{
    // Walking
    double A = -0.014;
    double w = M_PI * 3 / 2;
    double T = 2 * M_PI / w;
    double delta = T / 8;
    double t = fmod(tt, T); // 0 to T
    double t1, T1, w1, t3, T3, w3;
    double h = 0.005;
    double Kp_sup = 8, Kp_air = 5, Kd_sup = 0.08, Kd_air = 0.05;
    double Kp_lf_ssp[12] = {Kp_sup, Kp_sup, Kp_sup, Kp_sup, Kp_sup, Kp_sup,
                            Kp_air, Kp_air, Kp_air, Kp_air, Kp_air, Kp_air};
    double Kp_rf_ssp[12] = {Kp_air, Kp_air, Kp_air, Kp_air, Kp_air, Kp_air,
                            Kp_sup, Kp_sup, Kp_sup, Kp_sup, Kp_sup, Kp_sup};
    double Kd_lf_ssp[12] = {Kd_sup, Kd_sup, Kd_sup, Kd_sup, Kd_sup, Kd_sup,
                            Kd_air, Kd_air, Kd_air, Kd_air, Kd_air, Kd_air};
    double Kd_rf_ssp[12] = {Kd_air, Kd_air, Kd_air, Kd_air, Kd_air, Kd_air,
                            Kd_sup, Kd_sup, Kd_sup, Kd_sup, Kd_sup, Kd_sup};

    // The site positions
    double txyz_ip[3], lfxyz_ip[3], rfxyz_ip[3];
    // Both the feet and the torso retain constant orientation
    double tkphi_ip[4];
    mju_copy4(tkphi_ip, tkphi);
    double lfkphi_ip[4];
    mju_copy4(lfkphi_ip, lfkphi);
    double rfkphi_ip[4];
    mju_copy4(rfkphi_ip, rfkphi);

    // printf("t: %lg\n", t);

    if (t < (T / 4 - delta))
    {
        // DSP (1/2)
        txyz_ip[0] = A * sin(w * t) + txyz[0]; // torso
        txyz_ip[1] = 0 + txyz[1];              // torso
        txyz_ip[2] = 0 + txyz[2];              // torso
        lfxyz_ip[0] = 0 + lfxyz[0];            // left foot
        lfxyz_ip[1] = 0 + lfxyz[1];            // left foot
        lfxyz_ip[2] = 0 + lfxyz[2];            // left foot
        rfxyz_ip[0] = 0 + rfxyz[0];            // right foot
        rfxyz_ip[1] = 0 + rfxyz[1];            // right foot
        rfxyz_ip[2] = 0 + rfxyz[2];            // right foot

        bioloid_12dof_IK_position_v2(mm, 0, 1, 2, txyz_ip, tkphi_ip, lfxyz_ip,
                                     lfkphi_ip, rfxyz_ip, rfkphi_ip, qjans, 20,
                                     1e-6, 1e-5);
        controller_pd(mm, dd, qjans, dqjans, Kp_hold, Kd_hold, Tmax_hold, 7, 12);
    }
    else if (t < (T / 4 + delta))
    {
        // SSP right leg
        t1 = t - (T / 4 - delta);
        T1 = 4 * delta;
        w1 = 2 * M_PI / T1;
        txyz_ip[0] = A * sin(w * t) + txyz[0];     // Torso
        txyz_ip[1] = 0 + txyz[1];                  // Torso
        txyz_ip[2] = 0 + txyz[2];                  // Torso
        lfxyz_ip[0] = 0 + lfxyz[0];                // left foot
        lfxyz_ip[1] = 0 + lfxyz[1];                // left foot
        lfxyz_ip[2] = h * sin(t1 * w1) + lfxyz[2]; // left foot up
        rfxyz_ip[0] = 0 + rfxyz[0];                // right foot
        rfxyz_ip[1] = 0 + rfxyz[1];                // right foot
        rfxyz_ip[2] = 0 + rfxyz[2];                // right foot

        bioloid_12dof_IK_position_v2(mm, 0, 1, 2, txyz_ip, tkphi_ip, lfxyz_ip,
                                     lfkphi_ip, rfxyz_ip, rfkphi_ip, qjans, 20,
                                     1e-6, 1e-5);
        controller_pd(mm, dd, qjans, dqjans, Kp_hold, Kd_hold, Tmax_hold, 7, 12);
    }
    else if (t < (3 * T / 4 - delta))
    {
        // DSP
        txyz_ip[0] = A * sin(w * t) + txyz[0]; // torso
        txyz_ip[1] = 0 + txyz[1];              // torso
        txyz_ip[2] = 0 + txyz[2];              // torso
        lfxyz_ip[0] = 0 + lfxyz[0];            // left foot
        lfxyz_ip[1] = 0 + lfxyz[1];            // left foot
        lfxyz_ip[2] = 0 + lfxyz[2];            // left foot
        rfxyz_ip[0] = 0 + rfxyz[0];            // right foot
        rfxyz_ip[1] = 0 + rfxyz[1];            // right foot
        rfxyz_ip[2] = 0 + rfxyz[2];            // right foot

        bioloid_12dof_IK_position_v2(mm, 0, 1, 2, txyz_ip, tkphi_ip, lfxyz_ip,
                                     lfkphi_ip, rfxyz_ip, rfkphi_ip, qjans, 20,
                                     1e-6, 1e-5);
        controller_pd(mm, dd, qjans, dqjans, Kp_hold, Kd_hold, Tmax_hold, 7, 12);
    }
    else if (t < (3 * T / 4 + delta))
    {
        // SSP left leg
        t3 = t - (3 * T / 4 - delta);
        T3 = 4 * delta;
        w3 = 2 * M_PI / T3;
        txyz_ip[0] = A * sin(w * t) + txyz[0];     // Torso
        txyz_ip[1] = 0 + txyz[1];                  // Torso
        txyz_ip[2] = 0 + txyz[2];                  // Torso
        lfxyz_ip[0] = 0 + lfxyz[0];                // left foot
        lfxyz_ip[1] = 0 + lfxyz[1];                // left foot
        lfxyz_ip[2] = 0 + lfxyz[2];                // left foot
        rfxyz_ip[0] = 0 + rfxyz[0];                // right foot
        rfxyz_ip[1] = 0 + rfxyz[1];                // right foot
        rfxyz_ip[2] = h * sin(t3 * w3) + rfxyz[2]; // right foot up

        bioloid_12dof_IK_position_v2(mm, 0, 1, 2, txyz_ip, tkphi_ip, lfxyz_ip,
                                     lfkphi_ip, rfxyz_ip, rfkphi_ip, qjans, 20,
                                     1e-6, 1e-5);
        controller_pd(mm, dd, qjans, dqjans, Kp_hold, Kd_hold, Tmax_hold, 7, 12);
    }
    else
    {
        // DSP (2/2)
        txyz_ip[0] = A * sin(w * t) + txyz[0]; // Torso
        txyz_ip[1] = 0 + txyz[1];              // Torso
        txyz_ip[2] = 0 + txyz[2];              // Torso
        lfxyz_ip[0] = 0 + lfxyz[0];            // left foot
        lfxyz_ip[1] = 0 + lfxyz[1];            // left foot
        lfxyz_ip[2] = 0 + lfxyz[2];            // left foot
        rfxyz_ip[0] = 0 + rfxyz[0];            // right foot
        rfxyz_ip[1] = 0 + rfxyz[1];            // right foot
        rfxyz_ip[2] = 0 + rfxyz[2];            // right foot

        bioloid_12dof_IK_position_v2(mm, 0, 1, 2, txyz_ip, tkphi_ip, lfxyz_ip,
                                     lfkphi_ip, rfxyz_ip, rfkphi_ip, qjans, 20,
                                     1e-6, 1e-5);
        controller_pd(mm, dd, qjans, dqjans, Kp_hold, Kd_hold, Tmax_hold, 7, 12);
    }
}

void myCostFunc(const double dv_ip[12], double res_op[18], double jac_res_op[18 * 12], const double Amat[6 * 12], const mjData *dd)
{
    // Global variables: m_ID, d_ID
    // hi1();
    // fflush(stdout);

    int nsensordata = m_ID->nsensordata; // Need for accesing sensordata.

    mju_copy(d_ID->qacc + 6, dv_ip, 12); // set dv to qqacc+6
    mj_inverseSkip(m_ID, d_ID, mjSTAGE_VEL, /* don't skip sensor */ 0);

    // Residual [0-5]: tauB, the physics violation
    mju_copy(res_op, d_ID->qfrc_inverse, 6);
    // Scale
    double sqrt_W_tauB = mju_sqrt(W_tauB);
    mju_scl(res_op, res_op, sqrt_W_tauB, 6);

    double cfs[6], cfs_robot[6];

    for (int i = 0; i < 6; i++)
    {
        cfs[i] = d_ID->sensordata[cfs_idx[i]];
        cfs_robot[i] = dd->sensordata[cfs_idx[i]];
    }
    mju_sub(res_op + 6, cfs, cfs_robot, 6); // (cfs-cfs_robot) residual
    double sqrt_W_cfs = mju_sqrt(W_cfs);
    mju_scl(res_op + 6, res_op + 6, sqrt_W_cfs, 6); // Scale

    double acc_diff[12];
    mju_sub(acc_diff, dv_ip, dd->qacc + 6, 12);
    mju_mulMatVec(res_op + 6 + 6, Amat, acc_diff, 6, 12); // contact point acceleration residual
    double sqrt_W_pc_acc = mju_sqrt(W_pc_acc);
    mju_scl(res_op + 6 + 6, res_op + 6 + 6, sqrt_W_pc_acc, 6); // Scale

    // hi2();
    // fflush(stdout);
    /************************************************************************************/
    if (jac_res_op != nullptr)
    {
        double tmp[6 * 12];
        // DtauDaT: 18 x 18, DsenDaT: 18 x 2*(nsensordata) (contact sensor: found | force | pos (7))
        double DtauDaT[18 * 18], DsenDaT[18 * nsensordata]; // Outputs are transposed w.r.t. standard notation
        mjd_inverseFD(m_ID, d_ID, 1e-6, true, nullptr, nullptr, DtauDaT, nullptr, nullptr, DsenDaT, nullptr);
        // Take DtauDaT[6:18, 0:6], transpose them, and fill them as first 6 rows of jac_res_op (6 x 12)
        for (int i = 0; i < 6; i++)      // for each row
            for (int j = 0; j < 12; j++) // for each column
                tmp[i * 12 + j] = DtauDaT[(j + 6) * 18 + i];

        // Scale
        double W1[6 * 6];
        mju_eye(W1, 6);
        mju_scl(W1, W1, sqrt_W_tauB, 36);
        mju_mulMatMat(jac_res_op, W1, tmp, 6, 6, 12);

        // Take DsenDaT[6:18, cfs_idx(six indices)]. transpose them, and fill them as next 6 rows of jac_res_op (12 x 12)
        int idx = 0;
        for (int i = 0; i < 6; i++)
            for (int j = 0; j < 12; j++)
                tmp[i * 12 + j] = DsenDaT[(j + 6) * nsensordata + cfs_idx[i]];

        // Scale
        double W2[6 * 6];
        mju_eye(W2, 6);
        mju_scl(W2, W2, sqrt_W_cfs, 36);
        mju_mulMatMat(jac_res_op + 6 * 12, W2, tmp, 6, 6, 12);

        // Scaled Amat is the last 6 rows of jac_res_op (18 x 12)
        double W3[6 * 6];
        mju_eye(W3, 6);
        mju_scl(W3, W3, sqrt_W_pc_acc, 36);
        mju_mulMatMat(jac_res_op + 12 * 12, W3, Amat, 6, 6, 12);
        // mju_copy(jac_res_op + 12 * 12, Amat, 6 * 12);
    }
    // hi3();
    // fflush(stdout);
};

void controller_SS_OC(const mjModel *m, mjData *d, double err_axis_angle[4], double err_w[3], double des_qj[12], double des_qjdot[12])
{
    if (imu_fusion.load())
    {
        if (!d->ncon)
        {
            printf("No contact detected switching to PD controller!\n");
            imu_fusion.store(false);
        }
        int nState = mj_stateSize(m, stateSIG_ID);
        std::vector<mjtNum> statevec(nState);
        mj_getState(m, d, statevec.data(), stateSIG_ID);
        mj_setState(m_ID, d_ID, statevec.data(), stateSIG_ID);
        // Copy qacc manually
        mju_copy(d_ID->qacc, d->qacc, m_ID->nv);
        // Keep the acceleration same and propagate the state
        mj_inverse(m_ID, d_ID);



        // set desired FB acceleration
        double des_fb_acc[6] = {0.}; // We can use the linear acceleration from IMU
        double cmd_fb_acc[6] = {0.};

        // the orientation correction controller
        double Kp_ori_fb_acc = 4.5 * err_axis_angle[3];
        double Kd_ori_fb_acc = 4.5;

        // zero des_fb_acc 6x1, hence skipped the computation
        mju_scl3(cmd_fb_acc + 3, err_axis_angle, Kp_ori_fb_acc);
        mju_scl3(err_w, err_w, Kd_ori_fb_acc);
        mju_addTo3(cmd_fb_acc + 3, err_w);

        mju_copy(d_ID->qacc, cmd_fb_acc, 6); // set the required FB acc
        // Rest of the joint accelerations to be solved by optimisation problem

        std::vector<mjtNum> JacPstack(3 * 2 * m->nv);
        // Left foot
        int found = int(d->sensordata[sOfst + 0]);
        if (found)
        {
            int addr = 4;
            double pt_pos[3];
            mju_copy3(pt_pos, d->sensordata + sOfst + addr);
            mj_jac(m, d, JacPstack.data(), nullptr, pt_pos, 7); // LF is body 7
        }
        // Right foot
        found = int(d->sensordata[0 + sOfst + 7]);
        if (found)
        {
            int addr = 4 + 7;
            double pt_pos[3];
            mju_copy3(pt_pos, d->sensordata + sOfst + addr);
            mj_jac(m, d, JacPstack.data() + 3 * (m->nv), nullptr, pt_pos, 13); // RF is body 13
        }

        // Extract only the required part: JacP_joints = JacPstack[:, 6:]
        std::vector<mjtNum> JacP_joints(3 * 2 * (m->nv - 6));
        for (int i = 0; i < 3 * 2; i++)
            for (int j = 0; j < m->nv - 6; j++)
                JacP_joints[i * (m->nv - 6) + j] = JacPstack[i * m->nv + (6 + j)];

        // Remove dependent rows in JacP_joints, if any. TODO for later.
        // If rank deficient, then switch to PD control.

        // Initial value of the design variables: joint accelerations
        double dv[12];
        mju_copy(dv, d->qacc + 6, 12);

        // Solve the optimisation problem here.
        ceres::Problem problem;
        problem.AddResidualBlock(new CtrlCostFunc(JacP_joints.data(), d), nullptr, dv);

        ceres::Solver::Options options;
        options.minimizer_type = ceres::TRUST_REGION;
        options.trust_region_strategy_type = ceres::LEVENBERG_MARQUARDT;
        options.linear_solver_type = ceres::DENSE_QR;
        // options.function_tolerance = 1e-3;
        // options.parameter_tolerance = 1e-4;
        options.gradient_tolerance = 1e-5;
        options.max_num_iterations = 10;
        options.minimizer_progress_to_stdout = false;

        ceres::Solver::Summary summary;
        ceres::Solve(options, &problem, &summary);
        // std::cout << summary.BriefReport() << std::endl;

        // Get timing information
        double total_time = summary.total_time_in_seconds;
        double minimizer_time = summary.minimizer_time_in_seconds;

        // Check for success
        // Check termination reason
        // if (summary.termination_type == ceres::CONVERGENCE)
        // {
        //   std::cout << "converged" << std::endl;
        // }

        // Or check if it converged
        if (!summary.IsSolutionUsable())
        {
            std::cout << "Solution may not be reliable" << std::endl;
            // Switch to PD control
            imu_fusion.store(false);
            return;
        }

        // Get joint torques
        mju_copy(d_ID->qacc + 6, dv, 12);
        mj_inverse(m_ID, d_ID);
        // mju_printMat(d_ID->qfrc_inverse, 1, 6);

        double tauID[12], facID = 0.6, tauPD[12], facPD = 0.5;
        // Tune the factor based on location of gCOP.

        // Apply the torque from inverse dynamics control
        mju_scl(tauID, d_ID->qfrc_inverse + 6, facID, 12);

        // Small amount of PD control torque
        double err_jpos[12], derr_jpos[12];
        mju_sub(err_jpos, des_qj, d->qpos + 7, 12);
        mju_sub(derr_jpos, des_qjdot, d->qvel + 6, 12);

        double Kp_hold_minimal[12], Kd_hold_minimal[12];
        mju_scl(Kp_hold_minimal, Kp_hold, facPD, 12);
        mju_scl(Kd_hold_minimal, Kd_hold_2, facPD, 12);
        for (int i = 0; i < 12; i++)
        {
            tauPD[i] = Kp_hold_minimal[i] * err_jpos[i] + Kd_hold_minimal[i] * derr_jpos[i];
        }

        // Total control torque
        mju_add(d->ctrl, tauID, tauPD, 12);
    }
    else
    {
        // Shadow simulation only, apply PD control on the filtered joint angles
        controller_pd(m, d, des_qj, des_qjdot, Kp_hold, Kd_hold_2, Tmax_1, 7, 12);
    }
};

void controller_wrapper()
{
    if (d_sim->ncon) // Controller
    {
        switch (choosecontroller)
        {
        case 1:
            controller_pd(m, d_sim, qjans, dqjans, Kp_hold, Kd_hold, Tmax_hold, 7, 12);
            break;
        case 2:
            // Print the d_sim time and example_motion_start_time
            // printf("d_sim time: %lg, example_motion_start_time: %lg\n", d_sim->time, example_motion_start_time);
            controller_IK_pd_motionexample1(m, d_sim, (d_sim->time - example_motion_start_time), qjans, dqjans);
            break;
        case 3:
            controller_IK_pd_motionexample2(m, d_sim, (d_sim->time - example_motion_start_time), qjans, dqjans);
            break;
        case 4:
            controller_IK_pd_motionexample3(m, d_sim, (d_sim->time - example_motion_start_time), qjans, dqjans);
            break;
        case 5:
            controller_IK_pd_motionexample4(m, d_sim, (d_sim->time - example_motion_start_time), qjans, dqjans);
            break;
        case 6:
            controller_IK_pd_motionexample5(m, d_sim, (d_sim->time - example_motion_start_time), qjans, dqjans);
            break;
        case 7:
            controller_IK_pd_motionexample6(m, d_sim, (d_sim->time - example_motion_start_time), qjans, dqjans);
            break;
        case 8:
            controller_IK_pd_motionexample7(m, d_sim, (d_sim->time - example_motion_start_time));
            break;
        default:
            controller_pd(m, d_sim, qjans, dqjans, Kp_hold, Kd_hold, Tmax_hold, 7, 12);
        }
    }
    else
    {
        controller_pd(m, d_sim, qjans, dqjans, Kp_hold, Kd_hold, Tmax_hold, 7, 12);
    }
}
/**************************************************************************************/
// keyboard callback function, it uses global structure variables
void keyboard_cps(GLFWwindow *window, int key, int scancode, int act, int mods)
{
    if (act == GLFW_PRESS)
    {
        switch (key)
        {
        case GLFW_KEY_G:
            // Toggle filter flag
            filter_flag ^= true;
            if (filter_flag)
                printf("Filter applied.\n");
            else
                printf("Filter removed.\n");
            break;

        case GLFW_KEY_E:
            // Toggle through reconstruction modes
            recons_num = (recons_num + 1) % RECONS_Nmodes;
            break;

        case GLFW_KEY_RIGHT_ALT:
            // Toggle through figure pages
            fig_num = (fig_num + 1) % FIG_Npages;
            break;

        case GLFW_KEY_R:
            record_data_flag ^= true;
            if (record_data_flag)
            {
                printf("Recording started at %lg.\n", glfwGetTime());
            }
            else
                printf("Recording stopped.\n");
            break;

        case GLFW_KEY_1:
            // mtx_d_sim.lock();
            choosecontroller = 1;
            printf("Controller switched to 1: %d\n", choosecontroller);
            example_motion_start_time = d_sim->time;
            // mtx_d_sim.unlock();
            break;

        case GLFW_KEY_2:
            // mtx_d_sim.lock();
            choosecontroller = 2;
            printf("Controller switched to 2: %d\n", choosecontroller);
            example_motion_start_time = d_sim->time;
            // mtx_d_sim.unlock();
            break;

        case GLFW_KEY_3:
            choosecontroller = 3;
            printf("Controller switched to 3: %d\n", choosecontroller);
            example_motion_start_time = d_sim->time;
            break;

        case GLFW_KEY_4:
            choosecontroller = 4;
            printf("Controller switched to 4: %d\n", choosecontroller);
            example_motion_start_time = d_sim->time;
            break;

        case GLFW_KEY_5:
            choosecontroller = 5;
            printf("Controller switched to 5: %d\n", choosecontroller);
            example_motion_start_time = d_sim->time;
            break;

        case GLFW_KEY_6:
            choosecontroller = 6;
            printf("Controller switched to 6: %d\n", choosecontroller);
            example_motion_start_time = d_sim->time;
            break;

        case GLFW_KEY_7:
            choosecontroller = 7;
            printf("Controller switched to 7: %d\n", choosecontroller);
            example_motion_start_time = d_sim->time;
            break;

        case GLFW_KEY_8:
            choosecontroller = 8;
            printf("Controller switched to 8: %d\n", choosecontroller);
            example_motion_start_time = d_sim->time;
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

        case GLFW_KEY_I:
            // Toggle IMU fusion mode ON/OFF
            imu_fusion = !imu_fusion.load();
            break;

        case GLFW_KEY_W:
            // Toggle between kinematic reconstruction or shadow simulation
            kr_ss = !kr_ss.load();
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
            record_data_flag = false;
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
