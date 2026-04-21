/*
Check the residual and residual jacobian functions
*/

#include <iostream>
#include "../include/mujoco/mujoco.h"
#include "my_mjc_func_declarations.h"

#include <vector>
#include <ceres/ceres.h>
#include <glog/logging.h>

// function for optimisation
int cfs_idx[6] = {1, 2, 3, 8, 9, 10}; // 1:4 and 8:11
int ncfs = 6;                         // Two feet fx fy fz

void residual_vec(double dv_ip[12], double res_op[18], const mjModel *mm, const double Amat[6 * 12], const mjData *dd, mjData *dd_ID)
{
    mju_copy(dd_ID->qacc + 6, dv_ip, 12); // set dv to qqacc+6

    mj_inverseSkip(mm, dd_ID, mjSTAGE_VEL, /* don't skip sensor */ 0);
    mju_copy(res_op, dd_ID->qfrc_inverse, 6); // copy the physics violation (tauB) to the residual

    double cfs[6], cfs_robot[6];
    for (int i = 0; i < 6; i++)
    {
        cfs[i] = dd_ID->sensordata[cfs_idx[i]];
        cfs_robot[i] = dd->sensordata[cfs_idx[i]];
    }
    mju_sub(res_op + 6, cfs, cfs_robot, 6); // (cfs-cfs_robot) residual

    double acc_diff[12];
    mju_sub(acc_diff, dv_ip, dd->qacc + 6, 12);
    mju_mulMatVec(res_op + 6 + 6, Amat, acc_diff, 6, 12); // contact point acceleration residual
}

void residual_jac(double dv_ip[12], double jac_res_op[18 * 12], const mjModel *mm, const double Amat[6 * 12], const mjData *dd, mjData *dd_ID)
{
    mju_copy(dd_ID->qacc + 6, dv_ip, 12); // set dv to qqacc+6
    mj_inverseSkip(mm, dd_ID, mjSTAGE_VEL, /* skip sensor */ 0);

    // DtauDaT: 18 x 18, DsenDaT: 18 x 2*(7) (contact sensor: found | force | pos)
    double DtauDaT[18 * 18], DsenDaT[18 * 14]; // Outputs are transposed w.r.t. standard notation
    mjd_inverseFD(mm, dd_ID, 1e-6, true, nullptr, nullptr, DtauDaT, nullptr, nullptr, DsenDaT, nullptr);

    // Take DtauDaT[6:18, 0:6], transpose them, and fill them as first 6 rows of jac_res_op (6 x 12)
    for (int i = 0; i < 6; i++)      // for each row
        for (int j = 0; j < 12; j++) // for each column
            jac_res_op[i * 12 + j] = DtauDaT[(j + 6) * 18 + i];

    // Take DsenDaT[6:18, cfs_idx(six indices)]. transpose them, and fill them as next 6 rows of jac_res_op (12 x 12)
    int idx = 0;
    for (int i = 0; i < 6; i++)
        for (int j = 0; j < 12; j++)
            jac_res_op[(i + 6) * 12 + j] = DsenDaT[(j + 6) * 14 + cfs_idx[i]];

    // Amat is the last 6 rows of jac_res_op (18 x 12)
    // mju_printMat(Amat, 6, 12);
    mju_copy(jac_res_op + 12 * 12, Amat, 6 * 12);
}

void myCostFunc(double dv_ip[12], double res_op[18], double jac_res_op[18 * 12], const mjModel *mm, const double Amat[6 * 12], const mjData *dd, mjData *dd_ID)
{
    mju_copy(dd_ID->qacc + 6, dv_ip, 12); // set dv to qqacc+6
    mj_inverseSkip(mm, dd_ID, mjSTAGE_VEL, /* don't skip sensor */ 0);

    // Residual [0-5]: tauB, the physics violation
    mju_copy(res_op, dd_ID->qfrc_inverse, 6);

    double cfs[6], cfs_robot[6];
    for (int i = 0; i < 6; i++)
    {
        cfs[i] = dd_ID->sensordata[cfs_idx[i]];
        cfs_robot[i] = dd->sensordata[cfs_idx[i]];
    }
    mju_sub(res_op + 6, cfs, cfs_robot, 6); // (cfs-cfs_robot) residual

    double acc_diff[12];
    mju_sub(acc_diff, dv_ip, dd->qacc + 6, 12);
    mju_mulMatVec(res_op + 6 + 6, Amat, acc_diff, 6, 12); // contact point acceleration residual

    /************************************************************************************/
    if (jac_res_op != nullptr)
    {
        // DtauDaT: 18 x 18, DsenDaT: 18 x 2*(7) (contact sensor: found | force | pos)
        double DtauDaT[18 * 18], DsenDaT[18 * 14]; // Outputs are transposed w.r.t. standard notation
        mjd_inverseFD(mm, dd_ID, 1e-6, true, nullptr, nullptr, DtauDaT, nullptr, nullptr, DsenDaT, nullptr);
        // Take DtauDaT[6:18, 0:6], transpose them, and fill them as first 6 rows of jac_res_op (6 x 12)
        for (int i = 0; i < 6; i++)      // for each row
            for (int j = 0; j < 12; j++) // for each column
                jac_res_op[i * 12 + j] = DtauDaT[(j + 6) * 18 + i];

        // Take DsenDaT[6:18, cfs_idx(six indices)]. transpose them, and fill them as next 6 rows of jac_res_op (12 x 12)
        int idx = 0;
        for (int i = 0; i < 6; i++)
            for (int j = 0; j < 12; j++)
                jac_res_op[(i + 6) * 12 + j] = DsenDaT[(j + 6) * 14 + cfs_idx[i]];

        // Amat is the last 6 rows of jac_res_op (18 x 12)
        // mju_printMat(Amat, 6, 12);
        mju_copy(jac_res_op + 12 * 12, Amat, 6 * 12);
    }
}

// Single parameter block of 12 joint accelerations
class CtrlCostFunc : public ceres::SizedCostFunction<18, 12>
{
public:
    CtrlCostFunc(const mjModel *mm,
                 const double *Amat,
                 const mjData *dd,
                 mjData *dd_ID)
        : mm_(mm), Amat_(Amat), dd_(dd), dd_ID_(dd_ID) {}

    bool Evaluate(double const *const *parameters,
                  double *residuals,
                  double **jacobians) const override
    {
        // parameters[0] points to your single block of 12 joint accelerations
        const double *joint_accel = parameters[0];

        // Determine if Jacobian is needed
        double *jac_ptr = (jacobians != nullptr && jacobians[0] != nullptr) ? jacobians[0] : nullptr;

        // Single function call - computes both residuals and Jacobian (if needed)
        myCostFunc(const_cast<double *>(joint_accel),
                   residuals,
                   jac_ptr, // Pass nullptr if not needed
                   mm_,
                   Amat_,
                   dd_,
                   dd_ID_);

        return true;
    }

private:
    const mjModel *mm_;
    const double *Amat_;
    const mjData *dd_;
    mjData *dd_ID_;
};

int main()
{
    // Load model
    char model_path[1000] = "/home/venky/mujoco/model/bioloid_5/flat_ground_scene.xml";
    char error[1000] = "Could not load binary model";
    mjModel *m = mj_loadXML(model_path, NULL, error, 1000);
    if (!m)
    {
        mju_error("Load model error: %s", error);
        return -1;
    }
    else
    {
        std::cout << "Model loaded successfully!" << std::endl;
    }

    mjData *d = mj_makeData(m);
    mjData *d_ID = mj_makeData(m);

    // Load keyframe 0
    mj_resetDataKeyframe(m, d, 0);

    // Run the forward dynamics simulation for 10 steps
    for (int i = 0; i < 10; i++)
    {
        mj_step(m, d);
    }

    // // Print d.qpos
    // mju_printMat(d->qpos, 1, 19);
    // mju_printMat(d->qvel, 1, 18);
    // mju_printMat(d->qacc, 1, 18);

    int statespecID = mjSTATE_PHYSICS;
    int nStateVarID = mj_stateSize(m, statespecID);
    mjtNum *stateForID = (mjtNum *)mju_malloc(nStateVarID * sizeof(mjtNum));

    // Get state vector for ID
    mj_getState(m, d, stateForID, statespecID);
    // Set state vector to data for ID
    mj_setState(m, d_ID, stateForID, statespecID);
    // Set slightly different accelerations in d_ID (scaled by 1.1)
    mju_scl(d_ID->qacc, d->qacc, 1.1, m->nv);

    // Propagate changes in d_ID
    mj_inverse(m, d_ID);

    // mju_printMat(d_ID->qfrc_inverse, 1, 18);

    // Gains for virtual torsional spring-damper at the torso
    // double Kp_ori_fb_acc = 4.;
    // double Kd_ori_fb_acc = 0.2;
    // double Kp_pos_fb_acc = 0.; // Not used in this work
    // double Kd_pos_fb_acc = 0.; // Not used in this work

    // Commanded accelerations for the floating base (torso)
    // double cmd_fc_acc[6] = {0.};

    // Compute the Jacobian matrices for (equivalent) contact points
    double JacPstack[6 * 18];

    // Left foot
    bool lc_found = bool(d->sensordata[0]);
    if (lc_found)
        mj_jac(m, d, JacPstack, nullptr, d->sensordata + 4, LF_BODY);

    // mju_printMat(JacPstack, 3, 18);

    // Right foot
    bool rc_found = bool(d->sensordata[0 + 7]);
    if (rc_found)
        mj_jac(m, d, JacPstack + (3 * 18), nullptr, d->sensordata + (4 + 7), RF_BODY);

    // Contact point acc: Jb * qb_Acc + Jj * qj_Acc + JacDot * q_dot == 0 (constraint)

    // mju_printMat(JacPstack, 6, 18);

    double JjPstack[6 * 12] = {0.};
    // Copy all rows, 6 to 18 columns of JacPstack to JjPstack
    for (int i = 0; i < 6; i++)
        for (int j = 0; j < 12; j++)
            JjPstack[i * 12 + j] = JacPstack[i * 18 + (j + 6)];

    // mju_printMat(JjPstack, 6, 12);

    // Trial of residual_vec function
    double dv_trial[12];
    // mju_copy(dv_trial, d->qacc + 6, 12); // Initialize dv_trial to current qacc of joints
    mju_zero(dv_trial, 12); // Set dv_trial to zero
    double resVec[18];
    residual_vec(dv_trial, resVec, m, JjPstack, d, d_ID);
    std::cout << "Residual Vector:" << std::endl;
    mju_printMat(resVec, 1, 18);

    // Trial of residual_jac function
    double jacRes[18 * 12];
    m->opt.integrator = mjINT_EULER; // Use Euler to match inverse FD assumptions
    residual_jac(dv_trial, jacRes, m, JjPstack, d, d_ID);
    m->opt.integrator = mjINT_RK4; // Restore to RK4
    std::cout << "Residual Jacobian:" << std::endl;
    mju_printMat(jacRes, 18, 12);

    // Trial of myCostFunc function
    double resVec2[18];
    double jacRes2[18 * 12];
    m->opt.integrator = mjINT_EULER; // Use Euler to match inverse FD assumptions
    myCostFunc(dv_trial, resVec2, jacRes2, m, JjPstack, d, d_ID);
    m->opt.integrator = mjINT_RK4; // Restore to RK4
    std::cout << "Cost Function Residual Vector:" << std::endl;
    mju_printMat(resVec2, 1, 18);
    std::cout << "Cost Function Residual Jacobian:" << std::endl;
    mju_printMat(jacRes2, 18, 12);

    mju_free(stateForID);
    mj_deleteData(d);
    mj_deleteData(d_ID);
    mj_deleteModel(m);
}