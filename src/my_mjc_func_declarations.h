#include "../include/mujoco/mujoco.h"
#include <GLFW/glfw3.h>
#include <cstdio>

// Body numbers based on Bioloid_5 with three mocap bodies
enum
{
    TORSO_BODY = 1,
    LF_BODY = 7,
    RF_BODY = 13
};

// Geometry numbers of the contact spheres found from Bioloid_4 XML model
// enum
// {
//     CP_LSW_J2 = 9,
//     CP_LNW_J1 = 10,
//     CP_LNE_J0 = 11,
//     CP_LSE_J3 = 12,
//     CP_RSW_J0 = 19,
//     CP_RNW_J3 = 20,
//     CP_RNE_J2 = 21,
//     CP_RSE_J1 = 22
// };

// Site number based on Bioloid 4/5 XML model
enum
{
    TORSO_SITE = 0,
    LF_SITE = 1,
    LNE_J0_SITE = 2,
    LNW_J1_SITE = 3,
    LSW_J2_SITE = 4,
    LSE_J3_SITE = 5,
    RF_SITE = 6,
    RSW_J0_SITE = 7,
    RSE_J1_SITE = 8,
    RNE_J2_SITE = 9,
    RNW_J3_SITE = 10
};

// Address in d->userdata (double) for copying RobotDataToRead and RobotDataToWrite
enum
{
    ADDR_CDA = 0,
    ADDR_ROBOT_DATA_TO_READ = 80,
    ADDR_ROBOT_DATA_TO_SEND = 150,
    ADDR_ROBOT_DESIRED_Q = 180,
    ADDR_ROBOT_DESIRED_QDOT = 192,
};

/* Print hi1 for debugging */
void hi1();
/* Print hi2 for debugging */
void hi2();
/* Print hi3 for debugging */
void hi3();
/* Print hi4 for debugging */
void hi4();

/* Further sorting of contact data structure is stored in this */
const int NCONMAX_CDA = 12;
struct ContactDataAnalysis
{
    int group_start_IDs[NCONMAX_CDA];
    int ngroups;
    int ncon_in_group[NCONMAX_CDA];

    int w_group_start_IDs[NCONMAX_CDA];
    int nwgroups;
    int ncon_in_w_group[NCONMAX_CDA];

    int nconw;
    int world_con_IDs[NCONMAX_CDA];

    double lf_cop[3];
    double rf_cop[3];
    double gcop[3];
};

/* Contact analysis outputs:
group_start_IDs
ncon_in_group
ngroups
nconw
world_con_IDs
*/
void updateContactDataAnalysis(const mjModel *mm, mjData *dd, ContactDataAnalysis *cda);

/* Copy CDA to d->userdata */
void copyCDAToDoubleArray(double *arr_op, const struct ContactDataAnalysis *cda_ip);

/* Copy d->userdata to CDA */
void copyCDAFromDoubleArray(struct ContactDataAnalysis *cda_op, const double *arr_ip);

/*
Print the global location of the body frame
*/
void print_body_global_pos(const mjModel *mm, mjData *dd);

/* Calculate dimensions of an equimomental ellipoid with given density */
void mj_eqMomEllipsoid(mjtNum *inertia_mat_ip, mjtNum rho, mjtNum *semi_axes, mjtNum *prin_orient_quat);

/* Returns robot COM speed */
double my_mju_com_speed(const mjModel *mm, mjData *dd);

// Apply push to body_id at COM
void push(const mjModel *mm, mjData *dd, int body_id);
// Clear push applied to body_id
void push_clear(const mjModel *mm, mjData *dd, int body_id);
// Impulsive push
void impulsive_push(const mjModel *mm, mjData *dd, int body_id);

// keyboard callback
void keyboard(GLFWwindow *window, int key, int scancode, int act, int mods);
// keyboard callback for the bioloid IK utility
void bioloid_keyboard_IK_util(GLFWwindow *window, int key, int scancode, int act, int mods);
// keyboard callback for the kondo IK utility
void kondo_keyboard_IK_util(GLFWwindow *window, int key, int scancode, int act, int mods);

/* Cubic interpolation segment */
int mycubic(double ts, double te, double qs, double qe, double dqs, double dqe, double t, double op_q_dq[]);

/* Cubic interpolation segment for 3d vector
op_q_dq=[q1 q2 q3 | dq1 dq2 dq3] */
int mycubic3(double ts, double te, double qs[], double qe[], double dqs[], double dqe[], double t, double op_q_dq[]);

/* Finds distance between points p1 and p2 of dimenision dim*/
double mydist(const double p1[], const double p2[], const int dim);

/* Gives IDs of farthest pair of points in the polygon
Dependencies: mydist()
Input the dimensions- 2D/ 3D
Input the number of vertices of the polygon */
void farthest_points(const double polygon[], const int dim, const int nvert, int *op_p1_id, int *op_p2_id);

/* Gives out a normalized axis passing through the pair of farthest points
Dependencies: farthest_points() function*/
void farthest_points_axis(const double polygon[], const int dim, const int nvert, double op_t_axis[]);

/* Convex hull of a list of points
Dependencies: Boost Geometry Library
ip_multi_point2D: Input the list of contact points
[x1,y1,x2,y2,...,xn,yn]
ip_len: number of contact points
op_poly2D: The convex hull of the points input through ip_poly2D
send in an array of length 'ip_multi_point_2D' + 2
[xh1,yh1,xh2,yh2,...,xh1,yh1] (Last point == First point)
op_poly2D_len: 'n+1', where 'n' is the number of vertices of the convex hull
spa: support polygon area */
void cHull2D(double ip_multi_point2D[], int ip_len, double op_poly2D[], int *op_poly2D_len, double *spa);

/* 2D Convex hull of contact points
Depends on chull2D() */
void contact_cHull(const mjModel *mm, mjData *dd, double op_poly2D[], int *op_poly2D_len, double *s_p_a);

/*
Dependencies: none at the moment
Tells if the point is inside the ellipse
*/
bool check_interior(const double gcz[], const double el_cx, const double el_cy, const double el_a, const double el_b);

/* Body id of geom1 of contact with id: con_id
No safety while accessing d->contact[con_id] */
int conbody1(const mjModel *mm, mjData *dd, int con_id);

/* Body id of geom2 of contact with id: con_id
No safety while accessing d->contact[con_id] */
int conbody2(const mjModel *mm, mjData *dd, int con_id);

/* Returns true if the contact with id: con_id has atleast one body as worldbody
No safety while accessing d->contact[con_id]
Dependencies: conbody1() and conbody2() functions */
bool isconbodyworld(const mjModel *mm, mjData *dd, int con_id);

/* Number of contact with the worldbody
con_id_op give the list of contact ids
function returns the length of the list */
int nconworld(const mjModel *mm, mjData *dd, int con_id_op[]);

/* Dependencies: none
Returns the number of body pairs in contact and the number of contact points in each pair
grp_start_id_op: Array of indices i where new group starts. mjdata.contact[i]
ncon_in_a_grp_op: Number of contacts in each contact group
ngrp_op: Number of contact groups
*/
void contactsort(const mjModel *mm, mjData *dd, int grp_start_id_op[], int ncon_in_a_grp_op[], int *ngrp_op);

/* Dependencies: contactsort
Returns the number of body pairs in contact and the number of contact points in each pair only if one of the bodies is worldbody
Inputs: grp_start_id_ip, ncon_in_a_grp_ip, ngrp_ip (outputs of contactsort function)
Outputs: wgrp_start_id_op, ncon_in_w_grp_op, nwgrp_op (details of those contact groups where one of the bodies is worldbody)
*/
void contactsort_world(const mjModel *mm, mjData *dd, int wgrp_start_id_op[], int ncon_in_w_grp_op[], int *nwgrp_op);

/* Dependencies: contactsort
COP, CENtroid and contact force (in global inertial frame) computed for a body pair with given ID grpno
*/
void contactgroupdata(const mjModel *mm, mjData *dd, int grpno, mjtNum *Rcop, mjtNum *Rcen, mjtNum *TotalContactForce);

/* Dependencies: contactgroupdata
Compute generalized COP for two body-ground pairs */
void genCOP(const mjModel *mm, mjData *dd, double g_cop_op[], ContactDataAnalysis *cda);

/*
wgrpID_ip: ID of the contact group with one of the bodies as worldbody
footbodyID_op: gives out body ID of the foot in contact
footNormal_op: normal to the contact surface (always points up)
*fz_op: input pointer to the magniture of the normal force
cop_op: output foot COP
*/
void footCOP(const mjModel *mm, mjData *dd, ContactDataAnalysis *cda, int wgrpID_ip, int footBodyID_op, double footNormal_op[3], double *fz_op, double cop_op[]);

/* For polygonal 3D
Dependencies: footCOP()
*/
void genCOP1(const mjModel *mm, mjData *dd, ContactDataAnalysis *cda, double lf_cop_op[], double rf_cop_op[], double g_cop_op[]);

/* Dependencies: None
Print contact data: positions, forces etc.*/
void printcontactdata(mjModel *mm, mjData *dd);

/* Dependencies: None*/
double mapValue(double v, double vmin, double vmax, double op_min, double op_max);

/*
Dependencies: none.
Inputs:
The focal point (3D) of the projection (e.g., light source)
Array of points (3D) to be projected: [x1,y1,z1,x2,y2,z2,...,xn,yn,zn]
Number of points to be projected: n_points.
The z-coordinate of the output points: op_z.
Output:
Array of projected points: [x1,y1,op_Z,x2,y2,op_z,...,xn,yn,op_z]
*/
void proj_contact_points(double focal_point[], double input_points[], int n_points, double op_z, double op_points[]);

/* Returns error: power delivered by actuators - rate of energy change
uses d->userdata[10 to 19]*/
double my_mju_powerbalance(const mjModel *mm, mjData *dd);

/*********Drawing (decorative) objects in scene*********/
void drawEllipsoid(const mjModel *mm, mjData *dd, mjtNum pt[], mjtNum semi_axes[], mjtNum *quat, float rgba[], mjvScene *scene, const mjvOption *opt);
void drawPointSph(const mjModel *mm, mjData *dd, mjtNum pt[3], mjtNum radius, float rgba[], mjvScene *scene, const mjvOption *opt);
void drawPointBox(const mjModel *mm, mjData *dd, mjtNum pt[3], mjtNum dim, float rgba[], mjvScene *scene, const mjvOption *opt);
void drawPush(mjModel *mm, mjData *dd, mjvScene *scene, const mjvOption *opt, int body_id);
void drawWorldFrame(mjModel *mm, mjData *dd, mjvScene *scene, const mjvOption *opt);
void drawCOM(mjModel *mm, mjData *dd, mjvScene *scene, const mjvOption *opt);
void drawCOMproj(mjModel *mm, mjData *dd, mjvScene *scene, const mjvOption *opt);
void drawCOPcongrp(mjModel *mm, mjData *dd, int grpno, ContactDataAnalysis *cda, mjvScene *scene, const mjvOption *opt);
void drawCENcongrp(mjModel *mm, mjData *dd, int grpno, mjvScene *scene, const mjvOption *opt);
void drawLineSeg(mjModel *mm, mjData *dd, mjtNum p1[], mjtNum p2[], float rgba[], mjvScene *scene, const mjvOption *opt);
void drawLineFootCOPs(mjModel *mm, mjData *dd, int grpno1, int grpno2, mjvScene *scene, const mjvOption *opt);

/*
Draw a planar rectangle with centre at pc, normal along normal vector, and half-sides a and b.
*/
void drawPlaneRect(mjModel *mm, mjData *dd, mjtNum pc[], mjtNum normal[], mjtNum a, mjtNum b, float rgba[], mjvScene *scene, const mjvOption *opt);

/* Draws a polygon joining these points in that order
IMP Input: pts-- array of size (n+1)x3. Last point should be same as the first point
Input: n--- number of points */
void drawPolygon(const mjModel *mm, mjData *dd, float rgba[], mjvScene *scene, const mjvOption *opt, double pts[], int n);

/* Draw generalized COP for stepped (hfield) terrain
One robot leg with contact group grpno1 is on ground
another is on a step
Draw a line joining them and also the generalized COP*/
void drawGenCOP(mjModel *mm, mjData *dd, ContactDataAnalysis *cda, mjvScene *scene, const mjvOption *opt);

/* Dependencies: drawPolygon, contact_hull, gen COP
Draws contact convex hull in the scene at the height of Generalized COP */
void drawContactConvexHull(const mjModel *mm, mjData *dd, ContactDataAnalysis *cda, float rgba[], mjvScene *scene, const mjvOption *opt);

void drawScaledContactConvexHull(const mjModel *mm, mjData *dd, ContactDataAnalysis *cda, float rgba[], mjvScene *scene, const mjvOption *opt);
/*********Kinematics functions*********/

double wrap_angles(double angle);

/* Subtract quaternions, qb*quat(res) = qa.
 mju_negQuat(qneg, qb);
 mju_mulQuat(qdif, qneg, qa);
 */
void my_mju_relQuat(mjtNum *qdif, const mjtNum qa[4], const mjtNum qb[4]);
void my_mj_differentiateXPos(mjtNum *qvel, mjtNum dt, const mjtNum *qpos1, const mjtNum *qpos2);
void my_mju_quat2axisAngle(const mjtNum *quat, mjtNum *axisa);
void my_mju_mat2axisAngle(const mjtNum *mat, mjtNum *axisa);
void my_mju_avgRmat(const mjtNum *Rmat1, const mjtNum *Rmat2, mjtNum *op_Rmat);

/* New orientation of frame after (premultiplying) rotating it by k phi in space */
void my_mju_kphi_equiv(double incr_k_phi[], double curr_k_phi[], double op_k_phi[]);

/*********Inverse Kinematics*********/

/* Dependencies: Eigen library
tsno: torso site ID
lfsno: left foot site ID
rfsno: right foot site ID
global_torso: torso site desired XYZ position
global_torso_kphi: torso site desired orientation-- global coordinates axis: k and angle phi
same for left and right feet
qj_op: [qjl|qlr] 12x1
nitrmax: max iterations of the solver
ef: allowed norm error in task space
ex: allowed norm error in joint space */
void bioloid_12dof_IK_position(const mjModel *mm, int tsno, int lfsno, int rfsno, double global_torso[], double global_torso_kphi[], double global_leftfoot[], double global_leftfoot_kphi[], double global_rightfoot[], double global_rightfoot_kphi[], double qj_op[], int nitrmax, double ef, double ex);

void bioloid_12dof_IK_position_v2(const mjModel *mm, int tsno, int lfsno, int rfsno, double global_torso[], double global_torso_kphi[], double global_leftfoot[], double global_leftfoot_kphi[], double global_rightfoot[], double global_rightfoot_kphi[], double qj_op[], int nitrmax, double ef, double ex);

void bioloid_12dof_IK_position_v3(const mjModel *mm, int tsno, int lfsno, int rfsno,
                                  double global_torso[], double global_torso_kphi[],
                                  double global_leftfoot[], double global_leftfoot_kphi[],
                                  double global_rightfoot[], double global_rightfoot_kphi[],
                                  double qj_op[],
                                  int nitrmax, double ef, double ex);
/* Includes velocity IK */
void bioloid_12dof_IK(const mjModel *mm, int tsno, int lfsno, int rfsno, double global_torso[], double global_torso_kphi[], double global_leftfoot[], double global_leftfoot_kphi[], double global_rightfoot[], double global_rightfoot_kphi[], double qj_op[],
                      double global_torso_vel[], double global_torso_omega[], double global_leftfoot_vel[], double global_leftfoot_omega[], double global_rightfoot_vel[], double global_rightfoot_omega[], double dqj_op[], int nitrmax, double ef, double ex);

/* Dependencies: Eigen library
Regulate COM position + Torso orientation
tsno: torso site ID
lfsno: left foot site ID
rfsno: right foot site ID
global_com: centre of mass global position
global_torso: torso site desired XYZ position
global_torso_kphi: torso site desired orientation-- global coordinates axis: k and angle phi
same for left and right feet
qj_op: [qjl|qlr] 12x1
nitrmax: max iterations of the solver
ef: allowed norm error in task space
ex: allowed norm error in joint space */
void bioloid_12dof_CoM_IK_position(const mjModel *mm, int tsno, int lfsno, int rfsno, double global_com[], double global_torso[], double global_torso_kphi[], double global_leftfoot[], double global_leftfoot_kphi[], double global_rightfoot[], double global_rightfoot_kphi[], double qj_op[], int nitrmax, double ef, double ex);

void bioloid_12dof_CoM_IK(const mjModel *mm, int tsno, int lfsno, int rfsno, double global_com[], double global_torso[], double global_torso_kphi[], double global_leftfoot[], double global_leftfoot_kphi[], double global_rightfoot[], double global_rightfoot_kphi[], double qj_op[], double global_com_vel[], double global_torso_omega[], double global_leftfoot_vel[], double global_leftfoot_omega[], double global_rightfoot_vel[], double global_rightfoot_omega[], double dqj_op[], int nitrmax, double ef, double ex);

/* Dependencies: Eigen library
tsno: torso (waist) site ID
lfsno: left foot site ID
rfsno: right foot site ID
global_torso: torso (waist) site desired XYZ position
global_torso_kphi: torso (waist) site desired orientation-- global coordinates axis: k and angle phi
same for left and right feet
qj_op: [qjl|qlr| qjuseless] [6|6| 10]x1
nitrmax: max iterations of the solver
ef: allowed norm error in task space
ex: allowed norm error in joint space */
void kondo_12dof_IK_position(const mjModel *mm, int tsno, int lfsno, int rfsno, double global_torso[], double global_torso_kphi[], double global_leftfoot[], double global_leftfoot_kphi[], double global_rightfoot[], double global_rightfoot_kphi[], double qj_op[], int nitrmax, double ef, double ex);

void kondo_12dof_IK_position_v2(const mjModel *mm, int tsno, int lfsno, int rfsno, double global_torso[], double global_torso_kphi[], double global_leftfoot[], double global_leftfoot_kphi[], double global_rightfoot[], double global_rightfoot_kphi[], double qj_op[], int nitrmax, double ef, double ex);

/* Includes velocity IK */
void kondo_12dof_IK(const mjModel *mm, int tsno, int lfsno, int rfsno, double global_torso[], double global_torso_kphi[], double global_leftfoot[], double global_leftfoot_kphi[], double global_rightfoot[], double global_rightfoot_kphi[], double qj_op[],
                    double global_torso_vel[], double global_torso_omega[], double global_leftfoot_vel[], double global_leftfoot_omega[], double global_rightfoot_vel[], double global_rightfoot_omega[], double dqj_op[], int nitrmax, double ef, double ex);

/* Dependencies: Eigen library
Regulate COM position + Torso orientation
tsno: torso site ID
lfsno: left foot site ID
rfsno: right foot site ID
global_com: centre of mass global position
global_torso: torso site desired XYZ position
global_torso_kphi: torso site desired orientation-- global coordinates axis: k and angle phi
same for left and right feet
qj_op: [qjl|qlr] 12x1
nitrmax: max iterations of the solver
ef: allowed norm error in task space
ex: allowed norm error in joint space */
void kondo_12dof_CoM_IK_position(const mjModel *mm, int tsno, int lfsno, int rfsno, double global_com[], double global_torso[], double global_torso_kphi[], double global_leftfoot[], double global_leftfoot_kphi[], double global_rightfoot[], double global_rightfoot_kphi[], double qj_op[], int nitrmax, double ef, double ex);

void kondo_12dof_CoM_IK(const mjModel *mm, int tsno, int lfsno, int rfsno, double global_com[], double global_torso[], double global_torso_kphi[], double global_leftfoot[], double global_leftfoot_kphi[], double global_rightfoot[], double global_rightfoot_kphi[], double qj_op[], double global_com_vel[], double global_torso_omega[], double global_leftfoot_vel[], double global_leftfoot_omega[], double global_rightfoot_vel[], double global_rightfoot_omega[], double dqj_op[], int nitrmax, double ef, double ex);

/* Wrap angles to the domain [-pi pi]*/
double wrap_angles(double angle);

void deltaPaxispt(const mjModel *mm, mjData *dd, double z, double opzmp[]);

/* ZMP
This function computes a point on non-central axis of contact forces
Works only on flat/ stepped terrain that is horizontal
gives ZMP coordinates at height z */
void deltaCaxispt(const mjModel *mm, mjData *dd, double z, double opzmp[]);

void drawdeltaPaxis(mjModel *mm, mjData *dd, double z1, double z2, mjvScene *scene, const mjvOption *opt);

void drawdeltaCaxis(mjModel *mm, mjData *dd, double z1, double z2, mjvScene *scene, const mjvOption *opt);

/* Based on gyroscope sensor defined in the model */
void drawFootRotationAxis(const mjModel *mm, mjData *dd, mjvScene *scene, const mjvOption *opt);

/* Print the qpos (e.g., generated from IK utility) to file named "qpos_export.txt" */
void save_qpos(const mjModel *mm, mjData *dd);

/* Load the qpos to mjData from file named "qpos_export.txt" */
void load_qpos(const mjModel *mm, mjData *dd);

/* Print txyz, tkphi, lfxyz, lfkphi, rfxyz, rfkphi to file "site_export.txt" */
void save_sitepose(double ip_txyz[3], double ip_tkphi[4], double ip_lfxyz[3], double ip_lfkphi[4], double ip_rfxyz[3], double ip_rfkphi[4]);

/* Print com_pos global variable to file "com_export.txt" */
void save_com_pos(double ip_com_pos[3]);

/* Load txyz, tkphi, lfxyz, lfkphi, rfxyz, rfkphi to file "site_export.txt" */
void load_sitepose(double op_txyz[3], double op_tkphi[4], double op_lfxyz[3], double op_lfkphi[4], double op_rfxyz[3], double op_rfkphi[4]);

/* Load com_pos global variable from file "com_export.txt" */
void load_com_pos(double op_com_pos[3]);

/* Save the locations of the contact points. */
void save_contact_points(const mjModel *mm, mjData *dd);

/* Write (append) data vector to the file with "file_name"*/
void append_data_vector_to_file(const char *file_name, double data[], int len);

/* Write (append) time and data vector to the file with "file_name"*/
void append_time_data_vector_to_file(const char *file_name, double time, double data[], int len);

/* Edit this to save whatever data you want and call based on 'data_recording' latch R to start, 0 to end */
void record_data(const mjModel *mm, mjData *dd);
void record_tvec3_data(const mjModel *mm, mjData *dd, double data[], char *file_name);
void record_vec12_data(double data[], char *file_name);

/*--------MuJoCo Plotting-- mjvFigure---------*/

/* Plot three data streams: Initialize figure title, legend, and axis ranges */
void plot_d3t_init(mjvFigure *myfig, char *mytitle, char *legend1, char *legend2, char *legend3, double last_nsec, double yrange[]);
/* Plot two data streams: Initialize figure title, legend, and axis ranges */
void plot_d2t_init(mjvFigure *myfig, char *mytitle, char *legend1, char *legend2, double last_nsec, double yrange[]);
/* Plot single data stream: Initialize figure title, legend, and length of time history to display */
void plot_d1t_init(mjvFigure *myfig, char *mytitle, char *legend1, double last_nsec, double yrange[]);

/* Update three data streams */
void plot_d3t_update(mjvFigure *myfig, double last_nsec, double yrange[], double data1, double data2, double data3);
/* Update two data streams */
void plot_d2t_update(mjvFigure *myfig, double last_nsec, double yrange[], double data1, double data2);
/* Update single data stream */
void plot_d1t_update(mjvFigure *myfig, double last_nsec, double yrange[], double data1);

/* For interpoltaion */
template <typename T>
void my_linspace_2d(T p1xy[], T p2xy[], int npoints, T opxy[])
{
    T dx = (p2xy[0] - p1xy[0]) / (npoints - 1);
    T dy = (p2xy[1] - p1xy[1]) / (npoints - 1);

    for (int i = 0; i < npoints; i++)
    {
        opxy[i * 2] = p1xy[0] + dx * i;     // x-coordinate
        opxy[i * 2 + 1] = p1xy[1] + dy * i; // y-coordinate
    }
}

/* COP-ZMP plot */
void plot_COP_init(mjvFigure *myfig);
void plot_COP_update(mjvFigure *myfig, mjModel *mm, mjData *dd, ContactDataAnalysis *cda);
