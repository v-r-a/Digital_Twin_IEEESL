#include <cstdio>

#include "../include/mujoco/mujoco.h"
#include "my_mjc_func_declarations.h"
#include "rs485_bus_functions.h"
#include "exp_meas_proc_func.h"

// Takes data from RobotDataToRead and mjData corresponding to the experiment and updates the ExpContactDataAnalysis structure.
void updateExpContactDataAnalysis(const mjModel *mm, mjData *dd_exp, RobotDataToRead *readIn, ExpContactDataAnalysis *exp_cda)
{
    // double lf_fsr[12]; // normal forces in the world frame
    // double rf_fsr[12]; // normal forces in the world frame

    // double lf_COP[3]; // in the world frame
    // double rf_COP[3]; // in the world frame
    // double genCOP[3]; // in the world frame

    double n[3] = {0, 0, 1};

    // Compute left foot normal GRF in the world frame
    for (int i = 0; i < 4; i++) // loop over four corners
    {
        n[2] = readIn->lf_fsr[i];
        mju_mulMatVec(exp_cda->lf_GRFs + 3 * i, dd_exp->xmat + 9 * LF_BODY, n, 3, 3);
    }

    // Compute the right foot normal GRF in the world frame
    for (int i = 0; i < 4; i++) // loop over four corners
    {
        n[2] = readIn->rf_fsr[i];
        mju_mulMatVec(exp_cda->rf_GRFs + 3 * i, dd_exp->xmat + 9 * RF_BODY, n, 3, 3);
    }

    // Run exp_genCOP to update the foot COPs and the genCOP.
    // exp_genCOP(mm, dd_exp, readIn, exp_cda->lf_COP, exp_cda->rf_COP, exp_cda->gCOP);
    exp_genCOP1(mm, dd_exp, readIn, 0, 0, exp_cda->lf_COP, exp_cda->rf_COP, exp_cda->gCOP);
}

// This function checks if any of the FSRs on the left foot are in contact with the ground.
// A force cut off value is defined internally.
// If the force on any of the FSRs is above this cut-off, then the function returns true.
bool exp_chk_lf_gnd_con(RobotDataToRead *readIn)
{
    // Define contact cut-off at 0.008 kg (8 g).
    double fz_cut_off = 0.025;
    // Find if the load values at all the four corners are above the cut-off.
    double fz = readIn->lf_fsr[FSR_LNE_J0] + readIn->lf_fsr[FSR_LNW_J1] + readIn->lf_fsr[FSR_LSW_J2] + readIn->lf_fsr[FSR_LSE_J3];
    bool lf_gnd_con = true;
    if (fz < fz_cut_off)
    {
        lf_gnd_con = false;
        // printf("Left foot is in the air.");
    }
    return lf_gnd_con;
}

// This function checks if any of the FSRs on the right foot are in contact with the ground.
// A force cut off value is defined internally.
// If the force on any of the FSRs is above this cut-off, then the function returns true.
bool exp_chk_rf_gnd_con(RobotDataToRead *readIn)
{
    // Define contact cut-off at 0.008 kg (8 g).
    double fz_cut_off = 0.025;
    // Find if the load values at all the four corners are above the cut-off.
    double fz = readIn->rf_fsr[FSR_RSW_J0] + readIn->rf_fsr[FSR_RSE_J1] + readIn->rf_fsr[FSR_RNE_J2] + readIn->rf_fsr[FSR_RNW_J3];
    bool rf_gnd_con = true;
    if (fz < fz_cut_off)
    {
        rf_gnd_con = false;
        // printf("Right foot is in the air.");
    }
    return rf_gnd_con;
}

// This function computes the foot COP in the left foot body frame.
void exp_lf_cop(const mjModel *mm, RobotDataToRead *readIn, double cop[3])
{
    // Remove this later
    int CP_LNE_J0 = mj_name2id(mm, mjOBJ_GEOM, "LNE_J0");
    int CP_LNW_J1 = mj_name2id(mm, mjOBJ_GEOM, "LNW_J1");
    int CP_LSW_J2 = mj_name2id(mm, mjOBJ_GEOM, "LSW_J2");
    int CP_LSE_J3 = mj_name2id(mm, mjOBJ_GEOM, "LSE_J3");

    // Location of touch points in the foot body frame. These are constants, these computations can be avoided.
    double r0[3], r1[3], r2[3], r3[3];
    mju_copy3(r0, mm->geom_pos + 3 * CP_LNE_J0);
    mju_copy3(r1, mm->geom_pos + 3 * CP_LNW_J1);
    mju_copy3(r2, mm->geom_pos + 3 * CP_LSW_J2);
    mju_copy3(r3, mm->geom_pos + 3 * CP_LSE_J3);

    // Compute the COP in the left foot body frame.
    double fz0 = readIn->lf_fsr[FSR_LNE_J0];
    double fz1 = readIn->lf_fsr[FSR_LNW_J1];
    double fz2 = readIn->lf_fsr[FSR_LSW_J2];
    double fz3 = readIn->lf_fsr[FSR_LSE_J3];
    double fz = fz0 + fz1 + fz2 + fz3;
    // print fz
    // std::cout << "fz Left leg: " << fz << std::endl;

    cop[0] = (r0[0] * fz0 + r1[0] * fz1 + r2[0] * fz2 + r3[0] * fz3) / fz;
    cop[1] = (r0[1] * fz0 + r1[1] * fz1 + r2[1] * fz2 + r3[1] * fz3) / fz;
    // Z coordinate of the COP is equal to the Z-offset of left foot site w.r.t. the left foot body frame.
    cop[2] = mm->site_pos[3 * LF_SITE + 2];
}

// This function computes the foot COP in the right foot body frame.
void exp_rf_cop(const mjModel *mm, RobotDataToRead *readIn, double cop[3])
{
    // Remove this later
    int CP_RSW_J0 = mj_name2id(mm, mjOBJ_GEOM, "RSW_J0");
    int CP_RSE_J1 = mj_name2id(mm, mjOBJ_GEOM, "RSE_J1");
    int CP_RNE_J2 = mj_name2id(mm, mjOBJ_GEOM, "RNE_J2");
    int CP_RNW_J3 = mj_name2id(mm, mjOBJ_GEOM, "RNW_J3");
    // Find the location of the touch points w.r.t. the right foot body frame.
    double r0[3], r1[3], r2[3], r3[3];
    mju_copy3(r0, mm->geom_pos + 3 * CP_RSW_J0);
    mju_copy3(r1, mm->geom_pos + 3 * CP_RSE_J1);
    mju_copy3(r2, mm->geom_pos + 3 * CP_RNE_J2);
    mju_copy3(r3, mm->geom_pos + 3 * CP_RNW_J3);

    // Compute the COP in the right foot body frame.
    double fz0 = readIn->rf_fsr[FSR_RSW_J0];
    double fz1 = readIn->rf_fsr[FSR_RSE_J1];
    double fz2 = readIn->rf_fsr[FSR_RNE_J2];
    double fz3 = readIn->rf_fsr[FSR_RNW_J3];
    double fz = fz0 + fz1 + fz2 + fz3;
    // print fz
    // std::cout << "fz Right leg: " << fz << std::endl;

    cop[0] = (r0[0] * fz0 + r1[0] * fz1 + r2[0] * fz2 + r3[0] * fz3) / fz;
    cop[1] = (r0[1] * fz0 + r1[1] * fz1 + r2[1] * fz2 + r3[1] * fz3) / fz;
    // Z coordinate of the COP is equal to the Z-offset of right foot site w.r.t. the right foot body frame.
    cop[2] = mm->site_pos[3 * RF_SITE + 2];
}

// This function computes the genCOP in the world frame based on following inputs:
// mjModel mm
// mjData dd_exp: Now this mjData can be corresponding the actual robot or something else.
// RobotDataToRead readIn: This is the data read from the robot.
// Output: double op_COP_LF[3]: This is the COP of the left foot in the world frame.
// Output: double op_COP_RF[3]: This is the COP of the right foot in the world frame.
// Output: double op_genCOP[3]: This is the genCOP valid for horizontal stepped terrain, in the world frame.
void exp_genCOP(const mjModel *mm, mjData *dd_exp, RobotDataToRead *readIn, double op_COP_LF[3], double op_COP_RF[3], double op_genCOP[3])
{
    // Check for foot-ground contact.
    bool lf_con = exp_chk_lf_gnd_con(readIn);
    bool rf_con = exp_chk_rf_gnd_con(readIn);

    if (lf_con && rf_con)
    {
        // Double leg stance

        // Compute the foot COP in the left foot body frame.
        double lf_cop[3], temp[3] = {0., 0., 0.};
        exp_lf_cop(mm, readIn, lf_cop);
        // Compute the foot COP in the world frame.
        mju_rotVecQuat(temp, lf_cop, dd_exp->xquat + 4 * LF_BODY); // Get this vector in world frame: vec = cop_local - body frame origin.
        mju_add3(op_COP_LF, temp, dd_exp->xpos + 3 * LF_BODY);     // Shift the origin from body frame to the world frame.
        double lf_NR = readIn->lf_fsr[FSR_LNE_J0] + readIn->lf_fsr[FSR_LNW_J1] + readIn->lf_fsr[FSR_LSW_J2] + readIn->lf_fsr[FSR_LSE_J3];

        // Compute the foot COP in the right foot body frame.
        double rf_cop[3];
        exp_rf_cop(mm, readIn, rf_cop);
        // Compute the foot COP in the world frame.
        mju_rotVecQuat(temp, rf_cop, dd_exp->xquat + 4 * RF_BODY); // Get this vector in world frame: vec = cop_local - body frame origin.
        mju_add3(op_COP_RF, temp, dd_exp->xpos + 3 * RF_BODY);     // Shift the origin from body frame to the world frame.
        double rf_NR = readIn->rf_fsr[FSR_RSW_J0] + readIn->rf_fsr[FSR_RSE_J1] + readIn->rf_fsr[FSR_RNE_J2] + readIn->rf_fsr[FSR_RNW_J3];

        // Compute the genCOP in the world frame as the weighted average of the two foot COPs (in the world frame).
        op_genCOP[0] = (lf_NR * op_COP_LF[0] + rf_NR * op_COP_RF[0]) / (lf_NR + rf_NR);
        op_genCOP[1] = (lf_NR * op_COP_LF[1] + rf_NR * op_COP_RF[1]) / (lf_NR + rf_NR);
        op_genCOP[2] = (lf_NR * op_COP_LF[2] + rf_NR * op_COP_RF[2]) / (lf_NR + rf_NR);
    }
    else if (lf_con)
    {
        // Compute the foot COP in the left foot body frame.
        double lf_cop[3], temp[3] = {0., 0., 0.};
        exp_lf_cop(mm, readIn, lf_cop);
        // Compute the foot COP in the world frame.
        mju_rotVecQuat(temp, lf_cop, dd_exp->xquat + 4 * LF_BODY); // Get this vector in world frame: vec = cop_local - body frame origin.
        mju_add3(op_COP_LF, temp, dd_exp->xpos + 3 * LF_BODY);     // Shift the origin from body frame to the world frame.

        mju_copy3(op_genCOP, op_COP_LF); // genCOP is same as COP_LF.
        mju_copy3(op_COP_RF, op_COP_LF); // COP_RF is same as COP_LF. This is for plotting purposes.
    }
    else if (rf_con)
    {
        // Compute the foot COP in the right foot body frame.
        double rf_cop[3], temp[3] = {0., 0., 0.};
        exp_rf_cop(mm, readIn, rf_cop);
        // Compute the foot COP in the world frame.
        mju_rotVecQuat(temp, rf_cop, dd_exp->xquat + 4 * RF_BODY); // Get this vector in world frame: vec = cop_local - body frame origin.
        mju_add3(op_COP_RF, temp, dd_exp->xpos + 3 * RF_BODY);     // Shift the origin from body frame to the world frame.

        mju_copy3(op_genCOP, op_COP_RF); // genCOP is same as COP_RF.
        mju_copy3(op_COP_LF, op_COP_RF); // COP_LF is same as COP_RF. This is for plotting purposes.
    }
    else
    {
        // Robot is in the air.
        // printf("exp_genCOP function: The robot is in the air. Over. I repeat,\n");
        mju_zero3(op_COP_LF);
        mju_zero3(op_COP_RF);
        mju_zero3(op_genCOP);
    }
}

void drawExpCOPs(const mjModel *mm, mjData *dd_exp, RobotDataToRead *readIn, mjvScene *scn_exp, mjvOption *opt_exp)
{
    float myMAGENTA[4] = {1, 0, 1, 0.8};
    float myCYAN[4] = {0, 1, 1, 0.8};
    // Foot COPs measured
    double COP_LF[3] = {0, 0, 0}, COP_RF[3] = {0, 0, 0}, genCOPexp[3] = {0, 0, 0};

    exp_genCOP1(mm, dd_exp, readIn, 0, 0, COP_LF, COP_RF, genCOPexp);

    drawPointBox(mm, dd_exp, COP_RF, 0.005, myCYAN, scn_exp, opt_exp);
    drawPointBox(mm, dd_exp, COP_LF, 0.005, myCYAN, scn_exp, opt_exp);
    drawPointBox(mm, dd_exp, genCOPexp, 0.008, myMAGENTA, scn_exp, opt_exp);
}

// Computes the left foot COP in the global frame even if the foot is tilted.
// mjModel mm
// mjData dd_exp: Now this mjData can be corresponding the actual robot or something else.
// RobotDataToRead readIn: This is the data read from the robot.
// cp_flag is contact point flag. if 1 then dd_exp->contact data is used.
// cf clag is contact force flag. If 1 then mj_contactforce() is used, otherwise FSR data is used.
// Output: double COP[3]: This is the COP of the left foot in the world frame.
// Output: double nL[3]: This is the normal vector of the left foot in the world frame.
void exp_lf_cop1(const mjModel *mm, mjData *dd_exp, RobotDataToRead *readIn, int cp_flag, int cf_flag, double COP[3], double nL[], double *fzL)
{
    // Find the orientation of the left foot in the world frame.
    double z[3] = {0, 0, 1};
    mju_mulMatVec(nL, dd_exp->xmat + 9 * LF_BODY, z, 3, 3);

    // Find the location of the touch points in the body frame
    double rNE[3], rNW[3], rSW[3], rSE[3];
    if (cp_flag == 0)
    {
        mju_copy3(rNE, mm->site_pos + 3 * LNE_J0_SITE);
        mju_copy3(rNW, mm->site_pos + 3 * LNW_J1_SITE);
        mju_copy3(rSW, mm->site_pos + 3 * LSW_J2_SITE);
        mju_copy3(rSE, mm->site_pos + 3 * LSE_J3_SITE);
    }
    else
    {
        // Workout the logic
    }

    // Find the contact forces at the touch points in the body frame.
    double fNE[3], fNW[3], fSW[3], fSE[3];
    if (cf_flag == 1)
    {
        // Workout the logic
    }

    // Computer the COP in the global frame.
    if (cp_flag == 0 && cf_flag == 0)
    {
        double fzNE = readIn->lf_fsr[FSR_LNE_J0];
        double fzNW = readIn->lf_fsr[FSR_LNW_J1];
        double fzSW = readIn->lf_fsr[FSR_LSW_J2];
        double fzSE = readIn->lf_fsr[FSR_LSE_J3];
        double fz = fzNE + fzNW + fzSW + fzSE;
        *fzL = fz;

        // COP in the body frame:
        double cop[3];
        cop[0] = (rNE[0] * fzNE + rNW[0] * fzNW + rSW[0] * fzSW + rSE[0] * fzSE) / fz;
        cop[1] = (rNE[1] * fzNE + rNW[1] * fzNW + rSW[1] * fzSW + rSE[1] * fzSE) / fz;
        cop[2] = (rNE[2] * fzNE + rNW[2] * fzNW + rSW[2] * fzSW + rSE[2] * fzSE) / fz;

        // COP in the global frame:
        mju_mulMatVec(COP, dd_exp->xmat + 9 * LF_BODY, cop, 3, 3);
        mju_add3(COP, COP, dd_exp->xpos + 3 * LF_BODY);
    }
    else
    {
        // Workout the logic
    }
}

// Computes the right foot COP in the global frame even if the foot is tilted.
// mjModel mm
// mjData dd_exp: Now this mjData can be corresponding the actual robot or something else.
// RobotDataToRead readIn: This is the data read from the robot.
// cp_flag is contact point flag. if 1 then dd_exp->contact data is used.
// cf clag is contact force flag. If 1 then mj_contactforce() is used, otherwise FSR data is used.
// Output: double COP[3]: This is the COP of the right foot in the world frame.
// Output: double nR[3]: This is the normal vector of the right foot in the world frame.
void exp_rf_cop1(const mjModel *mm, mjData *dd_exp, RobotDataToRead *readIn, int cp_flag, int cf_flag, double COP[3], double nR[], double *fzR)
{
    // Find the orientation of the right foot in the world frame.
    double z[3] = {0, 0, 1};
    mju_mulMatVec(nR, dd_exp->xmat + 9 * RF_BODY, z, 3, 3);

    // Find the location of the touch points in the body frame
    double rSW[3], rSE[3], rNE[3], rNW[3];
    if (cp_flag == 0)
    {
        mju_copy3(rSW, mm->site_pos + 3 * RSW_J0_SITE);
        mju_copy3(rSE, mm->site_pos + 3 * RSE_J1_SITE);
        mju_copy3(rNE, mm->site_pos + 3 * RNE_J2_SITE);
        mju_copy3(rNW, mm->site_pos + 3 * RNW_J3_SITE);
    }
    else
    {
        // Workout the logic
    }

    // Find the contact forces at the touch points in the body frame.
    double fSW[3], fSE[3], fNE[3], fNW[3];
    if (cf_flag == 1)
    {
        // Workout the logic
    }

    // Computer the COP in the global frame.
    if (cp_flag == 0 && cf_flag == 0)
    {
        double fzSW = readIn->rf_fsr[FSR_RSW_J0];
        double fzSE = readIn->rf_fsr[FSR_RSE_J1];
        double fzNE = readIn->rf_fsr[FSR_RNE_J2];
        double fzNW = readIn->rf_fsr[FSR_RNW_J3];
        double fz = fzSW + fzSE + fzNE + fzNW;
        *fzR = fz;

        // COP in the body frame:
        double cop[3];
        cop[0] = (rSW[0] * fzSW + rSE[0] * fzSE + rNE[0] * fzNE + rNW[0] * fzNW) / fz;
        cop[1] = (rSW[1] * fzSW + rSE[1] * fzSE + rNE[1] * fzNE + rNW[1] * fzNW) / fz;
        cop[2] = (rSW[2] * fzSW + rSE[2] * fzSE + rNE[2] * fzNE + rNW[2] * fzNW) / fz;

        // COP in the global frame:
        mju_mulMatVec(COP, dd_exp->xmat + 9 * RF_BODY, cop, 3, 3);
        mju_add3(COP, COP, dd_exp->xpos + 3 * RF_BODY);
    }
    else
    {
        // Workout the logic
    }
}

// Computes the genCOP for a general polygonal 3D case | stairs | flat terrain
void exp_genCOP1(const mjModel *mm, mjData *dd_exp, RobotDataToRead *readIn, int cp_flag, int cf_flag, double op_COP_LF[3], double op_COP_RF[3], double op_gCOP[3])
{
    // Check for foot-ground contact.
    bool lf_con = exp_chk_lf_gnd_con(readIn);
    bool rf_con = exp_chk_rf_gnd_con(readIn);
    // printf("lf_con: %d, rf_con: %d\n", lf_con, rf_con);

    if (lf_con && rf_con)
    {
        // Double leg stance
        double NL[3], NR[3], fzL, fzR;
        exp_lf_cop1(mm, dd_exp, readIn, cf_flag, cp_flag, op_COP_LF, NL, &fzL);
        exp_rf_cop1(mm, dd_exp, readIn, cf_flag, cp_flag, op_COP_RF, NR, &fzR);

        // Check if the left foot and the right foot are parallel
        double parallel = mju_dot3(NL, NR);
        double tol_parallel = 0.9997;
        if (parallel > tol_parallel)
        {
            // The feet are parallel
            double Navg[3];
            mju_add3(Navg, NL, NR);
            mju_normalize3(Navg);

            // Compute the genCOP in the world frame as the weighted average of the two foot COPs (each in the world frame)
            op_gCOP[0] = (fzL * op_COP_LF[0] + fzR * op_COP_RF[0]) / (fzL + fzR);
            op_gCOP[1] = (fzL * op_COP_LF[1] + fzR * op_COP_RF[1]) / (fzL + fzR);
            op_gCOP[2] = (fzL * op_COP_LF[2] + fzR * op_COP_RF[2]) / (fzL + fzR);
        }
        else
        {
            // The feet are not parallel
            double u[3], Navg[3];
            mju_cross(u, NL, NR);
            mju_normalize3(u);

            // Normal to the average plane
            Navg[0] = fzL * NL[0] + fzR * NR[0];
            Navg[1] = fzL * NL[1] + fzR * NR[1];
            Navg[2] = fzL * NL[2] + fzR * NR[2];
            mju_normalize3(Navg);

            // Compute a point on the average plane (call it oL)
            double tL[3];
            mju_cross(tL, NL, u);
            mju_normalize3(tL);

            double cLmcR[3];
            mju_sub3(cLmcR, op_COP_LF, op_COP_RF);
            // Distance from the left foot COP to the foot of perpendicular on the common line between the planes
            double dL = -mju_dot3(cLmcR, NR) / mju_dot3(tL, NR);
            double oL[3];
            mju_scl3(tL, tL, dL);
            mju_add3(oL, op_COP_LF, tL);

            // Compute the gCOP on the average plane using (oL, Navg)
            double roLcL[3], roLcR[3];
            mju_sub3(roLcL, op_COP_LF, oL);
            mju_sub3(roLcR, op_COP_RF, oL);
            double mL_oL[3], mR_oL[3], m_oL[3], fL[3], fR[3];
            mju_scl3(fL, NL, fzL);
            mju_scl3(fR, NR, fzR);
            mju_cross(mL_oL, roLcL, fL);
            mju_cross(mR_oL, roLcR, fR);
            mju_add3(m_oL, mL_oL, mR_oL);
            double fLpfR[3];
            mju_add3(fLpfR, fL, fR);
            double num[3], den;
            mju_cross(num, Navg, m_oL);
            den = mju_dot3(Navg, fLpfR);
            den = 1 / den;
            double r_oL_c[3];
            mju_scl3(r_oL_c, num, den);

            // Get the coordinates of the gCOP in the world frame.
            mju_add3(op_gCOP, oL, r_oL_c);
        }
    }
    else if (lf_con)
    {
        double tmp1[3], tmp2;
        exp_lf_cop1(mm, dd_exp, readIn, cf_flag, cp_flag, op_COP_LF, tmp1, &tmp2);
        mju_copy3(op_COP_RF, dd_exp->site_xpos + 3 * RF_SITE); // COP_RF is placed at foot centre for plotting purposes.
        mju_copy3(op_gCOP, op_COP_LF);                         // gCOP is same as COP_LF.
    }
    else if (rf_con)
    {
        double tmp1[3], tmp2;
        exp_rf_cop1(mm, dd_exp, readIn, cf_flag, cp_flag, op_COP_RF, tmp1, &tmp2);
        mju_copy3(op_COP_LF, dd_exp->site_xpos + 3 * LF_SITE); // COP_LF is placed at foot centre for plotting purposes.
        mju_copy3(op_gCOP, op_COP_RF);                         // gCOP is same as COP_RF.
    }
    else
    {
        // Robot is in the air.
        // printf("exp_genCOP function: The robot is in the air. Over. I repeat,\n");
        mju_zero3(op_COP_LF);
        mju_zero3(op_COP_RF);
        mju_zero3(op_gCOP);
    }
}