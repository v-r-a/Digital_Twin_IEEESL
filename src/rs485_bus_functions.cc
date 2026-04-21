
#include "rs485_bus_functions.h"

/* Functions to send data */

/* Functions to analyse received packet */
int sign(double x)
{
    if (x > 0)
        return 1;
    else if (x < 0)
        return -1;
    else
        return 0;
}

// This function computes the true load on each FSR.
// Dependencies: None.
// int ip: 10-bit ADC reading.
// int fsr_no: FSR number. Left foot numbers: 0,1,2,3. Right foot numbers: 4,5,6,7.
double fsr_true_load(int ip, int fsr_no)
{
    // extern int vRef_today[9];              // 10-bit ADC Vref readings for both the feet and the force_stick.
    extern int fsr_noLoad_today[9];           // 10-bit ADC preloaded sensor readings for both the feet and the force_stick.
    double m = FSR_calibration_scale[fsr_no]; // Slope of the 10-bit reading vs. load curve (Unit: 1/kg)
    // int cT = vRef_today[fsr_no];           // vref = ADC reading at true zero load recorded just before the experiment.
    // cT stays the same for one foot. Assumption: cT was the same at the time of calibration.
    int c0 = fsr_noLoad_today[fsr_no]; // ADC reading due to the current preload due to the indenter. No external load.
    // double Lp = (c0 - cT) / m;             // Preload computed from the calibration line. Note that this has to be positive. (Unit: kg).
    // double L = (ip - cT) / m - Lp;         // Actual load (Unit: kg).
    double L = ((double)(ip - c0)) / m;     // Actual load (Unit: kg).
    L = (double)((int)(L * 10000)) / 10000; // Truncate the deciamals beyond fourth decimal place.
    // Return zero if L is negative.
    if (L < 0)
    {
        L = 0.;
        // printf("Negative load detected: %lg.\n", L);
    }

    return L;
}

// Decode the packet recevied from the left or right leg.
// Dependencies: fsr_true_load()
// Inputs: packet, length of the packet.
// Outputs: Updates the RobotDataToRead structure.
void leg_data_decoder(uint8_t packet[], uint8_t length, RobotDataToRead *ReadData)
{
    // Check the length of the packet
    if (length == 31)
    {
        // First three bytes in the packet are reserved for addresses and function code.
        // The data: (A0|A1|A2|A3|wx|wy|wz|q0|q1|q2|q3|ax|ay|az)
        int s[4];
        int16_t u, x, y, z;
        double s_op[4], quat_op[4], acc_op[3], w_op[3];

        // Scales
        const double quat_scale = (1.0 / (1 << 14)); // Unitless
        const double gyr_scale = (1 / 900.0);        // 1rps = 900 LSB
        const double acc_scale = (1 / 100.0);        // 1m/s^2 = 100 LSB

        // Convert an array of 2-byte (LB|HB) integers to an array of 4-byte integers.
        s[0] = (packet[3] | packet[4] << 8);
        s[1] = (packet[5] | packet[6] << 8);
        s[2] = (packet[7] | packet[8] << 8);
        s[3] = (packet[9] | packet[10] << 8);

        // angular velocity provided by gyro is in the IMU's body fixed frame
        x = ((int16_t)packet[11]) | (((int16_t)packet[12]) << 8);
        y = ((int16_t)packet[13]) | (((int16_t)packet[14]) << 8);
        z = ((int16_t)packet[15]) | (((int16_t)packet[16]) << 8);
        w_op[0] = ((double)x) * gyr_scale;
        w_op[1] = ((double)y) * gyr_scale;
        w_op[2] = ((double)z) * gyr_scale;

        /* Quaternion describing the orientation of the IMU w.r.t
            the ground frame. The Z axis depends on when orientation when
            IMU was turned ON (if using the IMU fusion mode of BNO055) */
        u = ((uint16_t)packet[17]) | (((uint16_t)packet[18]) << 8);
        x = ((uint16_t)packet[19]) | (((uint16_t)packet[20]) << 8);
        y = ((uint16_t)packet[21]) | (((uint16_t)packet[22]) << 8);
        z = ((uint16_t)packet[23]) | (((uint16_t)packet[24]) << 8);
        quat_op[0] = ((double)u) * quat_scale;
        quat_op[1] = ((double)x) * quat_scale;
        quat_op[2] = ((double)y) * quat_scale;
        quat_op[3] = ((double)z) * quat_scale;

        // Linear acceleration: the net acceleration after deducting the gravity
        x = ((int16_t)packet[25]) | (((int16_t)packet[26]) << 8);
        y = ((int16_t)packet[27]) | (((int16_t)packet[28]) << 8);
        z = ((int16_t)packet[29]) | (((int16_t)packet[30]) << 8);
        acc_op[0] = ((double)x) * acc_scale;
        acc_op[1] = ((double)y) * acc_scale;
        acc_op[2] = ((double)z) * acc_scale;

        if (packet[0] == L_add)
        {
            memcpy(ReadData->lf_quat, quat_op, sizeof(quat_op));
            memcpy(ReadData->lf_acc, acc_op, sizeof(acc_op));
            memcpy(ReadData->lf_w, w_op, sizeof(w_op)); // Convert w to rad/s later.
            // Convert 10-bit FSR value to actual load using the calibration function.
            for (int i = 0; i < 4; i++)
                ReadData->lf_fsr[i] = fsr_true_load(s[i], i); // Left foot ids: 0,1,2,3
        }
        else if (packet[0] == R_add)
        {
            memcpy(ReadData->rf_quat, quat_op, sizeof(quat_op));
            memcpy(ReadData->rf_acc, acc_op, sizeof(acc_op));
            memcpy(ReadData->rf_w, w_op, sizeof(w_op));
            // Convert 10-bit FSR value to actual load using the calibration function.
            for (int i = 0; i < 4; i++)
                ReadData->rf_fsr[i] = fsr_true_load(s[i], i + 4); // Right foot ids: 4,5,6,7
        }
    }
    else if (length == 8)
    {
        // First three bytes in the packet are reserved for addresses and function code.
        // The data: (A0|A1|A2|A3)
        int s[4];
        double s_op[4];
        // Convert an array of 2-byte (LB|HB) integers to an array of 4-byte integers.
        s[0] = (packet[3] | packet[4] << 8);
        s[1] = (packet[5] | packet[6] << 8);
        s[2] = (packet[7] | packet[8] << 8);
        s[3] = (packet[9] | packet[10] << 8);

        if (packet[0] == L_add)
        {
            // Convert 10-bit FSR value to actual load using the calibration function.
            for (int i = 0; i < 4; i++)
                ReadData->lf_fsr[i] = fsr_true_load(s[i], i); // Left foot ids: 0,1,2,3
        }
        else if (packet[0] == R_add)
        {
            // Convert 10-bit FSR value to actual load using the calibration function.
            for (int i = 0; i < 4; i++)
                ReadData->rf_fsr[i] = fsr_true_load(s[i], i + 4); // Right foot ids: 4,5,6,7
        }
    }
}

void torso_data_decoder(uint8_t packet[], RobotDataToRead *ReadData)
{
    // First three bytes in the packet are reserved for addresses and function code.
    // The data: (wx|wy|wz|q0|q1|q2|q3|ax|ay|az)
    int16_t u, x, y, z;

    // Scales
    const double quat_scale = (1.0 / (1 << 14)); // Unitless
    const double gyr_scale = (1 / 900.0);        // 1rps = 900 LSB
    const double acc_scale = (1 / 100.0);        // 1m/s^2 = 100 LSB

    // angular velocity provided by gyro is in the IMU's body fixed frame
    x = ((int16_t)packet[3]) | (((int16_t)packet[4]) << 8);
    y = ((int16_t)packet[5]) | (((int16_t)packet[6]) << 8);
    z = ((int16_t)packet[7]) | (((int16_t)packet[8]) << 8);
    ReadData->torso_w[0] = ((double)x) * gyr_scale;
    ReadData->torso_w[1] = ((double)y) * gyr_scale;
    ReadData->torso_w[2] = ((double)z) * gyr_scale;

    /* Quaternion describing the orientation of the IMU w.r.t
    the ground frame. The Z axis depends on when orientation when
    IMU was turned ON (if using the IMU fusion mode of BNO055) */
    u = ((uint16_t)packet[9]) | (((uint16_t)packet[10]) << 8);
    x = ((uint16_t)packet[11]) | (((uint16_t)packet[12]) << 8);
    y = ((uint16_t)packet[13]) | (((uint16_t)packet[14]) << 8);
    z = ((uint16_t)packet[15]) | (((uint16_t)packet[16]) << 8);
    ReadData->torso_quat[0] = ((double)u) * quat_scale;
    ReadData->torso_quat[1] = ((double)x) * quat_scale;
    ReadData->torso_quat[2] = ((double)y) * quat_scale;
    ReadData->torso_quat[3] = ((double)z) * quat_scale;

    // Linear acceleration: the net acceleration after deducting the gravity
    x = ((int16_t)packet[17]) | (((int16_t)packet[18]) << 8);
    y = ((int16_t)packet[19]) | (((int16_t)packet[20]) << 8);
    z = ((int16_t)packet[21]) | (((int16_t)packet[22]) << 8);
    ReadData->torso_acc[0] = ((double)x) * acc_scale;
    ReadData->torso_acc[1] = ((double)y) * acc_scale;
    ReadData->torso_acc[2] = ((double)z) * acc_scale;
}

void actuator_data_decoder(uint8_t packet[], RobotDataToRead *ReadData)
{
    // Print the packet:
    // printf("Actuator packet: ");
    // for (int i = 0; i < 51; i++)
    //     printf("%d ", packet[i]);
    // printf("\n");

    // First three elements of the packet are reserved for addresses and function code.
    int ptr = 3;
    uint16_t tempq[12], tempqdot[12];
    // Read the data: th1LB | th1HB | dth1LB | dthHB ... th12LB | th12HB | dth12LB | dth12HB
    for (int i = 0; i < 12; i++)
    {
        // Build the Present Position value from LB | HB.
        tempq[i] = (packet[ptr] | packet[ptr + 1] << 8);
        // Build the Present Speed value from LB | HB.
        tempqdot[i] = (packet[ptr + 2] | packet[ptr + 3] << 8);
        // Increment
        ptr += 4;
    }
    // Print tempq:
    // printf("RE: ");
    // for (int i = 0; i < 12; i++)
    //     printf("%d ", tempq[i]);
    // printf("\n");

    // Convert the present angles and angular velocities to MuJoCo XML model convention.
    map_angles_robot_to_mjc(tempq, ReadData->q);
    map_angvel_robot_to_mjc(tempqdot, ReadData->qdot);
}

void fs_data_decoder(uint8_t packet[], FSDataToRead *FSData)
{

    // First three bytes in the packet are reserved for addresses and function code.
    // The data: (A2|q0|q1|q2|q3)
    int s;
    int16_t u, x, y, z;
    double s_op, quat_op[4];
    // Scales
    const double quat_scale = (1.0 / (1 << 14)); // Unitless

    // Convert an array of 2-byte (LB|HB) integers to an array of 4-byte integers.
    s = (packet[3] | packet[4] << 8);
    // printf("FSR value: %d\n", s);

    // Convert 10-bit FSR value to actual load using the calibration function.
    s_op = fsr_true_load(s, FSR_FS);

    u = ((uint16_t)packet[5]) | (((uint16_t)packet[6]) << 8);
    x = ((uint16_t)packet[7]) | (((uint16_t)packet[8]) << 8);
    y = ((uint16_t)packet[9]) | (((uint16_t)packet[10]) << 8);
    z = ((uint16_t)packet[11]) | (((uint16_t)packet[12]) << 8);
    quat_op[0] = ((double)u) * quat_scale;
    quat_op[1] = ((double)x) * quat_scale;
    quat_op[2] = ((double)y) * quat_scale;
    quat_op[3] = ((double)z) * quat_scale;

    FSData->fsr = s_op;
    memcpy(FSData->fs_quat, quat_op, sizeof(quat_op));

    // Print in a single row the fs_quat
    // printf("FS quat: ");
    // for (int i = 0; i < 4; i++)
    //     printf("%f ", FSData->fs_quat[i]);
    // printf("\n");
}

void map_angles_mjc_to_robot(double in[], uint16_t out[])
{
    // Take in joint angles in MuJoCo model and give out the corresponding motor motion angles
    // Order: [0 LHY,1 LHR,2 LHP,3 LKP,4 LAP,5 LAR, 6 RHY,7 RHR,8 RHP,9 RKP,10 RAP,11 RAR]
    // Scaling + Offset + Clipping (send warning if motion going out of limits)
    // 1023 = 300 degree rotation
    double scale = (1023.0 / 300.0) * (180.0 / M_PI);

    // Hip yaw
    double lhy = scale * in[0] + (double)HY_NOM;
    double rhy = scale * in[6] + (double)HY_NOM;
    out[0] = (uint16_t)lhy;
    out[6] = (uint16_t)rhy;
    if (lhy < HY_MIN || lhy > HY_MAX)
        printf("LHY out of range: %u \n", out[0]);
    if (rhy < HY_MIN || rhy > HY_MAX)
        printf("RHY out of range: %u \n", out[6]);

    // Hip roll
    double lhr = (-1) * scale * in[1] + (double)HR_NOM;
    double rhr = (-1) * scale * in[7] + (double)HR_NOM;
    out[1] = (uint16_t)lhr;
    out[7] = (uint16_t)rhr;
    if (lhr < HR_MIN || lhr > HR_MAX)
        printf("LHR out of range: %u \n", out[1]);
    if (rhr < HR_MIN || rhr > HR_MAX)
        printf("RHR out of range: %u \n", out[7]);

    // Hip pitch
    double lhp = scale * in[2] + (double)HP_NOM;
    double rhp = scale * in[8] + (double)HP_NOM;
    out[2] = (uint16_t)lhp;
    out[8] = (uint16_t)rhp;
    if (lhp < HP_MIN || lhp > HP_MAX)
        printf("LHP out of range: %u \n", out[2]);
    if (rhp < HP_MIN || rhp > HP_MAX)
        printf("RHP out of range: %u \n", out[8]);

    // Knee pitch
    double lkp = (-1) * scale * in[3] + (double)KP_NOM;
    double rkp = (-1) * scale * in[9] + (double)KP_NOM;
    out[3] = (uint16_t)lkp;
    out[9] = (uint16_t)rkp;
    if (lkp < KP_MIN || lkp > KP_MAX)
        printf("LKP out of range: %u \n", out[3]);
    if (rkp < KP_MIN || rkp > KP_MAX)
        printf("RKP out of range: %u \n", out[9]);

    // Ankle pitch
    double lap = (-1) * scale * in[4] + (double)AP_NOM;
    double rap = (-1) * scale * in[10] + (double)AP_NOM;
    out[4] = (uint16_t)lap;
    out[10] = (uint16_t)rap;
    if (lap < AP_MIN || lap > AP_MAX)
        printf("LAP out of range: %u \n", out[4]);
    if (rap < AP_MIN || rap > AP_MAX)
        printf("RAP out of range: %u \n", out[10]);

    // Ankle roll
    double lar = scale * in[5] + (double)AR_NOM;
    double rar = scale * in[11] + (double)AR_NOM;
    out[5] = (uint16_t)lar;
    out[11] = (uint16_t)rar;
    if (lar < AR_MIN || lar > AR_MAX)
        printf("LAR out of range: %u \n", out[5]);
    if (rar < AR_MIN || rar > AR_MAX)
        printf("RAR out of range: %u \n", out[11]);

    // Optional: double printout[12];
    // for (int i = 0; i < 12; i++)
    //     printout[i] = (double)out[i];
    // mju_printMat(printout, 1, 12);
}

void map_angles_robot_to_mjc(uint16_t in[], double out[])
{
    // Take in joint angles from the motors and give out the joint angles of the MuJoCo model
    // Order: [0 LHY,1 LHR,2 LHP,3 LKP,4 LAP,5 LAR, 6 RHY,7 RHR,8 RHP,9 RKP,10 RAP,11 RAR]
    // Scaling + Offset
    // 1023 = 300 degree rotation
    double scale = (300.0 / 1023.0) * (M_PI / 180.0);

    // Hip yaw
    out[0] = scale * ((double)in[0] - (double)HY_NOM);
    out[6] = scale * ((double)in[6] - (double)HY_NOM);

    // Hip roll
    out[1] = (-1.0) * scale * ((double)in[1] - (double)HR_NOM);
    out[7] = (-1.0) * scale * ((double)in[7] - (double)HR_NOM);

    // Hip pitch
    out[2] = scale * ((double)in[2] - (double)HP_NOM);
    out[8] = scale * ((double)in[8] - (double)HP_NOM);

    // Knee pitch
    out[3] = (-1.0) * scale * ((double)in[3] - (double)KP_NOM);
    out[9] = (-1.0) * scale * ((double)in[9] - (double)KP_NOM);

    // Ankle pitch
    out[4] = (-1.0) * scale * ((double)in[4] - (double)AP_NOM);
    out[10] = (-1.0) * scale * ((double)in[10] - (double)AP_NOM);

    // Ankle roll
    out[5] = scale * ((double)in[5] - (double)AR_NOM);
    out[11] = scale * ((double)in[11] - (double)AR_NOM);
}

uint16_t map_ang_speed_to_motor(double w_plus)
{

    // Zero angular speed has special meaning in Dynamixel. We cannot have the motors moving at full speed.
    if (abs(w_plus) < 0.1) // 0.05 --> 4 (10-bit)
    {
        w_plus = w_plus * 1.5 + 0.1; // 0.1--> 8 (10-bit)
        // printf("Zero angular speed detected.\n");
    }

    // Convert w (angular speed) from rad/s to RPM
    double rpm = w_plus * 60. / (2. * M_PI);
    // AX-12A RPM max 59
    if (abs(rpm) > 100)
    {
        printf("RPM out of range.\n");
        return 0; // Setting 0 => max speed to reach the Goal Position.
    }

    // Convert the value to 10-bit resolution. 1023 = 114 RPM
    double val = rpm * 1023. / 114.;

    // 0-1023 is CCW, 1024-2047 is CW rotation
    if (val > 0)
    {
        return (uint16_t)(val);
    }
    else if (val < 0)
    {
        printf("Negative moving speed not allowed.\n");
        return 1; // Slow speed towards the Goal position.
    }
    else
    {
        return 0;
    }
}

double conv_ang_vel_from_motor(uint16_t val_10bit)
{
    // Present speed can be positive or negative like angular velocity.
    // AX-12A: 0-1023 is CCW, 1024-2047 is CW rotation
    // Output in rad/s with sign.
    if (val_10bit > uint16_t(1023))
    {
        return -((double)val_10bit - 1024.) * (114. / 1023.) * (2. * M_PI / 60.);
    }
    else
    {
        return ((double)val_10bit) * (114. / 1023.) * (2. * M_PI / 60.);
    }
}

void map_angvel_mjc_to_robot(double in[], uint16_t out[])
{
    // Scaling has been taken care of in map_ang_speed_to_motor
    // Take care of sign of angular velocity here. (Sign convention to match map_angles_mjc_to_robot)

    // convert all inputs to positive and generate output.
    double temp[12];
    for (int i = 0; i < 12; i++)
    {
        temp[i] = abs(in[i]);
        out[i] = map_ang_speed_to_motor(temp[i]);
    }
    // printf("Vel: ");
    // for (int i = 0; i < 12; i++)
    //     printf("%u ", out[i]);
    // printf("\n");
}

void map_angvel_robot_to_mjc(uint16_t in[], double out[])
{
    // Map the joint axes orientation here. Reference: map_angles_bioloid_to_mjc().
    // Dependency: conv_ang_vel_from_Dynamixel()
    // Order: [0 LHY,1 LHR,2 LHP,3 LKP,4 LAP,5 LAR, 6 RHY,7 RHR,8 RHP,9 RKP,10 RAP,11 RAR]

    // Hip yaw
    out[0] = conv_ang_vel_from_motor(in[0]);
    out[6] = conv_ang_vel_from_motor(in[6]);
    // Hip roll
    out[1] = (-1) * conv_ang_vel_from_motor(in[1]);
    out[7] = (-1) * conv_ang_vel_from_motor(in[7]);
    // Hip pitch
    out[2] = conv_ang_vel_from_motor(in[2]);
    out[8] = conv_ang_vel_from_motor(in[8]);
    // Knee pitch
    out[3] = (-1) * conv_ang_vel_from_motor(in[3]);
    out[9] = (-1) * conv_ang_vel_from_motor(in[9]);
    // Ankle pitch
    out[4] = (-1) * conv_ang_vel_from_motor(in[4]);
    out[10] = (-1) * conv_ang_vel_from_motor(in[10]);
    // Ankle roll
    out[5] = conv_ang_vel_from_motor(in[5]);
    out[11] = conv_ang_vel_from_motor(in[11]);
}

int packet_send_builder(uint8_t packet[], RobotDataToSend *writeData)
{
    // This packet is broadcasted to all the nodes in the robot.

    packet[0] = PC_add;           // From
    packet[1] = All_add;          // To
    packet[2] = Req_default_data; // Function code

    uint16_t q_temp[12], qdot_temp[12];
    map_angles_mjc_to_robot(writeData->q, q_temp);
    // Print q_temp array
    // printf("PC : ");
    // for (int i = 0; i < 12; i++)
    //     printf("%d ", q_temp[i]);
    // printf("\n");
    map_angvel_mjc_to_robot(writeData->qdot, qdot_temp);
    for (int i = 0; i < 12; i++)
    {
        // th1 | th1_dot | th2 | th2_dot | ... | th12 | th12_dot
        packet[3 + 4 * i] = DXL_LOBYTE(q_temp[i]);
        packet[3 + 4 * i + 1] = DXL_HIBYTE(q_temp[i]);
        packet[3 + 4 * i + 2] = DXL_LOBYTE(qdot_temp[i]);
        packet[3 + 4 * i + 3] = DXL_HIBYTE(qdot_temp[i]);
    }

    int packet_length = 51;
    return packet_length;
}

void packet_received_analyser(uint8_t packet[], uint8_t packet_size, RobotDataToRead *ReadData)
{
    // This function analyses the received packet and updates the ReadData structure.
    uint8_t from_add = packet[0];
    uint8_t to_add = packet[1];
    // uint8_t function_code = packet[2];

    if (to_add == PC_add)
    {
        switch (from_add)
        {
        case T_add:
            torso_data_decoder(packet, ReadData);
            // printf("T\n");
            break;
        case L_add:
            leg_data_decoder(packet, packet_size, ReadData);
            // printf("L\n");
            break;

        case R_add:
            leg_data_decoder(packet, packet_size, ReadData);
            // printf("R\n");
            break;

        case A_add:
            actuator_data_decoder(packet, ReadData);
            // printf("A\n");
            break;
        }
    }
    else
    {
        printf(" This packet is addressed to %u, not the PC.\n", to_add);
    }
}