#include <stdint.h>
#include <stdio.h>
#include <math.h>
#include <string.h>

#ifndef DXL_LOBYTE
#define DXL_LOBYTE(w) ((uint8_t)(((uint64_t)(w)) & 0xff))
#endif

#ifndef DXL_HIBYTE
#define DXL_HIBYTE(w) ((uint8_t)((((uint64_t)(w)) >> 8) & 0xff))
#endif

#define DEVICENAME "/dev/ttyUSB0"
#define BAUDRATE 1000000

// The following two quantities: FSR_ref and FSR_slope define the calibration line of the FSRs.
// FSR calibration line: intercept L0-L1-L2-L3-R0-R1-R2-R3-FS
// Useless at the moment
// const double FSR_calibration_cutoff[9] = {281.51,
//                                           258.40,
//                                           202.75,
//                                           269.18,
//                                           280.07,
//                                           252.88,
//                                           231.74,
//                                           272.85,
//                                           227};

// FSR calibration line: slope (10-bit value) /kg (L0-L1-L2-L3-R0-R1-R2-R3-FS)
const double FSR_calibration_scale[9] = {1198.8,
                                         1230.1,
                                         1296.9,
                                         1231.0,
                                         1000., // 1047.5,
                                         1333.9,
                                         1355.4,
                                         1231.4,
                                         1500.0};

// FSR reading at zero load may vary due to the spring preload. So the following values are recorded before every experiment.
// FSR calibration line: current intercept (ideally same as FSR_ref)
// double FSR_ref_zero[9] = {300,
//                           300,
//                           300,
//                           300,
//                           300,
//                           300,
//                           300,
//                           300,
//                           300};

enum
{                   // Index: Left foot: 0-3, Force stick: 8.
    FSR_LNE_J0 = 0, // Left foot, North-East (J0)
    FSR_LNW_J1 = 1, // Left foot, North-West (J1)
    FSR_LSW_J2 = 2, // Left foot, South-West (J2)
    FSR_LSE_J3 = 3, // Left foot, South-East (J3)

    FSR_FS = 8 // Force stick (J2)
};

enum
{
    // Index: Left foot: 0-3, Right foot: 4-7, Force stick: 8.
    FSR_RSW_J0 = 0, // Right foot, South-West (J0)
    FSR_RSE_J1 = 1, // Right foot, South-East (J1)
    FSR_RNE_J2 = 2, // Right foot, North-East (J2)
    FSR_RNW_J3 = 3  // Right foot, North-West (J3)
};

/* C enum listing the nominal, minimum and maximum values of motor positions */
enum
{
    HY_MIN = 620,
    HY_MAX = 720,
    HY_NOM = 666,
    HY_INI = 666,

    HR_MIN = 440, // 472,
    HR_MAX = 584, // 552,
    HR_NOM = 512,
    HR_INI = 512,

    HP_MIN = 490,
    HP_MAX = 780,
    HP_NOM = 563, // 563.2,
    HP_INI = 563,

    KP_MIN = 560,
    KP_MAX = 915, // old 885
    KP_NOM = 614, // 614.4,
    KP_INI = 614,

    AP_MIN = 300,
    AP_MAX = 500,
    AP_NOM = 461, // 460.8,
    AP_INI = 450,

    AR_MIN = 440, // 472,
    AR_MAX = 584, // 552,
    AR_NOM = 512,
    AR_INI = 512
};

// C enum listing all the time outs and sleep times in microseconds.
enum
{
    RS485_TO = 15900,           // Reply time out in microseconds.
    SLEEP_BETWEEN_CALLS = 5000, // Slowdown the communication.
    LOOP_TIME = 19800,          // Loop time in microseconds.
    DATA_OUT_SLEEP = 100        // Sleep until data moves out. 1 byte needs 10us @ 1Mbps.
};

// C++ enum listing all the microcontroller addresses.
enum : uint8_t
{
    PC_add = 255,
    All_add = 10,
    A_add = 1,
    T_add = 2,
    L_add = 3,
    R_add = 4,
    FS_add = 5
};

/* C++ enum listing all the function codes */
enum : uint8_t
{
    Req_all_data = 1,
    Req_fsr_data = 2,
    Req_imu_data = 3,
    Req_actuator_data = 4,
    Sending_all_data = 5,
    Sending_fsr_data = 6,
    Sending_imu_data = 7,
    Sending_actuator_data = 8,
    Req_no_data = 9,
    Sending_no_data = 10,
    RW_actuator_data = 11,
    Req_default_data = 12,
    Req_fsr_init_data = 13,
    Sending_fsr_init_data = 14,
    W_actuator_data = 15,
    Req_imu_cal_data = 16,
    Sending_imu_cal_data = 17
};

// structure definition: read data from force stick.
struct FSDataToRead
{
    /* data */
    double fsr;
    double fs_quat[4];
};

// structure definition: read incomming sensor data.
struct RobotDataToRead
{
    /* data */
    double torso_quat[4];
    double lf_quat[4];
    double rf_quat[4];

    double torso_acc[3];
    double lf_acc[3];
    double rf_acc[3];

    double torso_w[3];
    double lf_w[3];
    double rf_w[3];

    double lf_fsr[4]; // Saved in order J0-J1-J2-J3
    double rf_fsr[4]; // Saved in order J0-J1-J2-J3

    double q[12];
    double qdot[12];
};

const int SIZE_FILTER_WINDOW = 5;

// structure definition: write to robot actuators.
struct RobotDataToSend
{
    /* Desired angle and instantaneous angular velocity */
    double q[12];
    double qdot[12];

    // Compliance slope can be added, i.e., Kp.
};

int sign(double x);

// Angle and angular velocity mapping functions

/* Convert joint angles from the MuJoCo model to the Bioloid motor angles.
 * Depencencies: None.
 * Inputs: in[]: array of joint angles from the MuJoCo model.
 * Outputs: out[]: array of motor angles (10-bit) for the Bioloid.
 */
void map_angles_mjc_to_robot(double in[], uint16_t out[]);

/* Convert joint angles read from the motor encoders to the MuJoCo model angles.
 * Depencencies: None.
 * Inputs: in[]: array of joint angles (10-bit) from the Bioloid.
 * Outputs: out[]: array of joint angles for the MuJoCo model.
 */
void map_angles_robot_to_mjc(uint16_t in[], double out[]);

/* Internal function.
 * Convert absolute moving speed from rad/s to a 10-bit value for the Dynamixel motors.
 * Depencencies: None.
 * Negative input not allowed
 */
uint16_t map_ang_speed_to_motor(double w);

/* Internal function.
 * Read angular velocity from Dynamixel motor.
 * Depencencies: None.
 * Inputs: val: 10-bit value read from the Dynamixel motor.
 * Outputs: (Signed) angular velocity in rad/s.
 */
double conv_ang_vel_from_motor(uint16_t val);

/* Convert the joint angle velocities from the MuJoCo model to the Bioloid motor speeds.
 * Take into account axis directions and scaling.
 * Depencencies: map_ang_speed_to_Dynamixel().
 * Inputs: in[]: array of joint angle velocities from the MuJoCo model.
 * Outputs: out[]: array of motor speeds (10-bit) for the Bioloid motors.
 * Note that Motor Speeds to be set can only be positive.
 */
void map_angvel_mjc_to_robot(double in[], uint16_t out[]);

/* Convert the joint angle velocities read from the Bioloid motors to the MuJoCo model.
 * Take into account axis directions and scaling.
 * Depencencies: read_ang_vel_from_Dynamixel().
 * Inputs: in[]: array of joint angle velocities (10-bit) from the Bioloid motors.
 * Outputs: out[]: array of joint angle velocities for the MuJoCo model.
 */
void map_angvel_robot_to_mjc(uint16_t in[], double out[]);

// Communication interpretation functions
double fsr_true_load(int ip, int fsr_no);

void fs_data_decoder(uint8_t packet[], FSDataToRead *FSData);

void actuator_data_decoder(uint8_t packet[], RobotDataToRead *ReadData);

void leg_data_decoder(uint8_t packet[], uint8_t length, RobotDataToRead *ReadData);

void torso_data_decoder(uint8_t packet[], RobotDataToRead *ReadData);

int packet_send_builder(uint8_t packet[], RobotDataToSend *writeData);

void packet_received_analyser(uint8_t packet[], uint8_t packet_size, RobotDataToRead *ReadData);
