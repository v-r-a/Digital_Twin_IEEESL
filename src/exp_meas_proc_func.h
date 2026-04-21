#include <cstdio>

#include "../include/mujoco/mujoco.h"

/* Processing the input of foot IMU and FSRs */
struct ExpContactDataAnalysis
{
    double lf_GRFs[12]; // normal forces in the world frame
    double rf_GRFs[12]; // normal forces in the world frame

    double lf_COP[3]; // in the world frame
    double rf_COP[3]; // in the world frame
    double gCOP[3]; // in the world frame
};

void updateExpContactDataAnalysis(const mjModel *mm, mjData *dd_exp, RobotDataToRead *readIn, ExpContactDataAnalysis *exp_cda);

void exp_lf_cop(const mjModel *mm, RobotDataToRead *readIn, double cop[3]);

void exp_rf_cop(const mjModel *mm, RobotDataToRead *readIn, double cop[3]);

void exp_genCOP(const mjModel *mm, mjData *dd_exp, RobotDataToRead *readIn, double op_COP_LF[3], double op_COP_RF[3], double op_genCOP[3]);

void exp_lf_cop1(const mjModel *mm, mjData *dd_exp, RobotDataToRead *readIn, int cp_flag, int cf_flag, double COP[3], double nL[], double *fzL);

void exp_rf_cop1(const mjModel *mm, mjData *dd_exp, RobotDataToRead *readIn, int cp_flag, int cf_flag, double COP[3], double nR[], double *fzR);

void exp_genCOP1(const mjModel *mm, mjData *dd_exp, RobotDataToRead *readIn, int cp_flag, int cf_flag, double op_COP_LF[3], double op_COP_RF[3], double op_gCOP[3]);

void drawExpCOPs(const mjModel *mm, mjData *dd, RobotDataToRead *readIn, mjvScene *scn, mjvOption *opt);
