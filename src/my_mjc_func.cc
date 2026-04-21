
#include <cstdio>
#include <iostream>

#include <eigen3/Eigen/Dense>
#include <eigen3/Eigen/Eigenvalues>

#include <boost/geometry.hpp>
#include <boost/geometry/geometries/multi_point.hpp>
#include <boost/geometry/geometries/polygon.hpp>

#include "../include/mujoco/mujoco.h"
#include <GLFW/glfw3.h>
#include "my_mjc_func_declarations.h"

/***************Structure related functions***************/

void updateContactDataAnalysis(const mjModel *mm, mjData *dd, ContactDataAnalysis *cda)
{
    if (dd->ncon)
    {
        contactsort(mm, dd, cda->group_start_IDs, cda->ncon_in_group, &(cda->ngroups));
        contactsort_world(mm, dd, cda->w_group_start_IDs, cda->ncon_in_w_group, &(cda->nwgroups));
        cda->nconw = nconworld(mm, dd, cda->world_con_IDs);

        // Compute the generalised cop using genCOP function only if there are any contacts with the world.
        if (cda->nconw)
            genCOP1(mm, dd, cda, cda->lf_cop, cda->rf_cop, cda->gcop);
    }
    else
    {
        cda->ngroups = 0;
        cda->nconw = 0;
        cda->nwgroups = 0;

        cda->gcop[0] = 0.;
        cda->gcop[1] = 0.;
        cda->gcop[2] = 0.;
    }
    // mju_printMat(cda->lf_cop, 1, 3);
    // mju_printMat(cda->rf_cop, 1, 3);
    // mju_printMat(cda->gcop, 1, 3);
}

void copyCDAToDoubleArray(double *arr_op, const struct ContactDataAnalysis *cda_ip)
{
    int i, j = 0;
    for (i = 0; i < NCONMAX_CDA; ++i)
    {
        arr_op[j++] = (double)cda_ip->group_start_IDs[i];
    }
    arr_op[j++] = (double)cda_ip->ngroups;
    for (i = 0; i < NCONMAX_CDA; ++i)
    {
        arr_op[j++] = (double)cda_ip->ncon_in_group[i];
    }
    arr_op[j++] = (double)cda_ip->nconw;
    for (i = 0; i < NCONMAX_CDA; ++i)
    {
        arr_op[j++] = (double)cda_ip->world_con_IDs[i];
    }
    for (i = 0; i < 3; ++i)
    {
        arr_op[j++] = cda_ip->gcop[i];
    }
    for (i = 0; i < 3; ++i)
    {
        arr_op[j++] = cda_ip->lf_cop[i];
    }
    for (i = 0; i < 3; ++i)
    {
        arr_op[j++] = cda_ip->rf_cop[i];
    }
}

void copyCDAFromDoubleArray(struct ContactDataAnalysis *cda_op, const double *arr_ip)
{
    int i, j = 0;
    for (i = 0; i < NCONMAX_CDA; ++i)
    {
        cda_op->group_start_IDs[i] = (int)arr_ip[j++];
    }
    cda_op->ngroups = (int)arr_ip[j++];
    for (i = 0; i < NCONMAX_CDA; ++i)
    {
        cda_op->ncon_in_group[i] = (int)arr_ip[j++];
    }
    cda_op->nconw = (int)arr_ip[j++];
    for (i = 0; i < NCONMAX_CDA; ++i)
    {
        cda_op->world_con_IDs[i] = (int)arr_ip[j++];
    }
    for (i = 0; i < 3; ++i)
    {
        cda_op->gcop[i] = arr_ip[j++];
    }
    for (i = 0; i < 3; ++i)
    {
        cda_op->lf_cop[i] = arr_ip[j++];
    }
    for (i = 0; i < 3; ++i)
    {
        cda_op->rf_cop[i] = arr_ip[j++];
    }
}

/***************Function definitions***************/

void hi1()
{
    printf("hi 1\n");
}

void hi2()
{
    printf("hi 2\n");
}

void hi3()
{
    printf("hi 3\n");
}

void hi4()
{
    printf("hi 4\n");
}

void print_body_global_pos(const mjModel *mm, mjData *dd)
{
    // Make sure mj_forward() is run before this function call.

    // Compute body frame location and print it
    for (int i = 0; i < mm->nbody; i++)
    {
        printf("Body %d global position: %lg %lg %lg\n", i, dd->xpos[3 * i], dd->xpos[3 * i + 1], dd->xpos[3 * i + 2]);
    }
}

void mj_eqMomEllipsoid(mjtNum *inertia_mat_ip, mjtNum rho, mjtNum *semi_axes, mjtNum *prin_orient_quat)
{
    mjtNum eigval[3], eigvec[9];
    // get the orientation of the principal axes of the inertia ellipsoid: prin_orient_quat
    mju_eig3(eigval, eigvec, prin_orient_quat, inertia_mat_ip);

    // calculate the semi-axes of the ellipsoid. Based on Lee & Goswami 2007, ICRA
    mjtNum cons = 15 / (8 * M_PI * rho);
    cons = mju_pow(cons, 1.0 / 5.0);

    mjtNum t1, t2, t3;
    t1 = (-eigval[0] + eigval[1] + eigval[2]);
    t2 = (eigval[0] - eigval[1] + eigval[2]);
    t3 = (eigval[0] + eigval[1] - eigval[2]);

    semi_axes[0] = cons * mju_pow(t1, 2.0 / 5.0) / mju_pow(t2 * t3, 1.0 / 10.0);
    semi_axes[1] = cons * mju_pow(t2, 2.0 / 5.0) / mju_pow(t1 * t3, 1.0 / 10.0);
    semi_axes[2] = cons * mju_pow(t3, 2.0 / 5.0) / mju_pow(t1 * t2, 1.0 / 10.0);
}

void angvel_from_angmom(double angmom3x1[], double inertia3x3[], double op_w[])
{

    /*     Eigen::Map<Eigen::Matrix<double, 3, 3, Eigen::RowMajor == 1>> I_eig(inertia3x3, 3, 3);
        Eigen::Map<Eigen::VectorXd> h_eig(angmom3x1, 3);
        Eigen::Map<Eigen::VectorXd> w_eig(op_w, 3);
        w_eig = I_eig.llt().solve(h_eig); */

    // w = inv(I)*h
    double inertia[9];
    double rank;
    mju_copy(inertia, inertia3x3, 9);
    mju_cholFactor(inertia, 3, rank);
    mju_cholSolve(op_w, inertia, angmom3x1, 3);
}

double my_mju_com_speed(const mjModel *mm, mjData *dd)
{
    double com_vel[3];
    mju_copy3(com_vel, dd->subtree_linvel);

    return mju_norm3(com_vel);
}

int mycubic(double ts, double te, double qs, double qe, double dqs, double dqe, double t, double op_q_dq[])
{
    double d = qs;
    double c = dqs;
    double b = (3 * qe - 3 * qs - (dqe + 2 * dqs) * (te - ts)) / ((te - ts) * (te - ts));
    double a = (2 * qs - 2 * qe + (dqe + dqs) * (te - ts)) / ((te - ts) * (te - ts) * (te - ts));

    // q(t)
    double dt = (t - ts);
    op_q_dq[0] = a * dt * dt * dt + b * dt * dt + c * dt + d;
    // qdot(t)
    op_q_dq[1] = 3 * a * dt * dt + 2 * b * dt + c;

    int flag = 1;
    // Sent warning if time is outside the cubic segment
    if (dt > (te - ts) || dt < 0)
        flag = 0;

    return flag;
}

int mycubic3(double ts, double te, double qs[], double qe[], double dqs[], double dqe[], double t, double op_q_dq[])
{
    // qs[] = qs1 qs2 qs3 and similarly for all others
    int flag = 1;
    // Sent warning if time is outside the cubic segment
    double dt = (t - ts);
    if (dt > (te - ts) || dt < 0)
        flag = 0;

    /*--------------------------------------------------------------*/
    double d = qs[0];
    double c = dqs[0];
    double b = (3 * qe[0] - 3 * qs[0] - (dqe[0] + 2 * dqs[0]) * (te - ts)) / ((te - ts) * (te - ts));
    double a = (2 * qs[0] - 2 * qe[0] + (dqe[0] + dqs[0]) * (te - ts)) / ((te - ts) * (te - ts) * (te - ts));

    // q(t)
    op_q_dq[0] = a * dt * dt * dt + b * dt * dt + c * dt + d;
    // qdot(t)
    op_q_dq[3] = 3 * a * dt * dt + 2 * b * dt + c;
    /*--------------------------------------------------------------*/
    d = qs[1];
    c = dqs[1];
    b = (3 * qe[1] - 3 * qs[1] - (dqe[1] + 2 * dqs[1]) * (te - ts)) / ((te - ts) * (te - ts));
    a = (2 * qs[1] - 2 * qe[1] + (dqe[1] + dqs[1]) * (te - ts)) / ((te - ts) * (te - ts) * (te - ts));

    // q(t)
    op_q_dq[1] = a * dt * dt * dt + b * dt * dt + c * dt + d;
    // qdot(t)
    op_q_dq[4] = 3 * a * dt * dt + 2 * b * dt + c;
    /*--------------------------------------------------------------*/
    d = qs[2];
    c = dqs[2];
    b = (3 * qe[2] - 3 * qs[2] - (dqe[2] + 2 * dqs[2]) * (te - ts)) / ((te - ts) * (te - ts));
    a = (2 * qs[2] - 2 * qe[2] + (dqe[2] + dqs[2]) * (te - ts)) / ((te - ts) * (te - ts) * (te - ts));

    // q(t)
    op_q_dq[2] = a * dt * dt * dt + b * dt * dt + c * dt + d;
    // qdot(t)
    op_q_dq[5] = 3 * a * dt * dt + 2 * b * dt + c;
    /*--------------------------------------------------------------*/

    return flag;
}

int conbody1(const mjModel *mm, mjData *dd, int con_id)
{
    return mm->geom_bodyid[dd->contact[con_id].geom1];
    // return mm->geom_bodyid[dd->contact[con_id].geom[0]]; // geom1 geom2 depricated
}

int conbody2(const mjModel *mm, mjData *dd, int con_id)
{
    return mm->geom_bodyid[dd->contact[con_id].geom2];
    // return mm->geom_bodyid[dd->contact[con_id].geom[1]]; // geom1 geom2 depricated
}

bool isconbodyworld(const mjModel *mm, mjData *dd, int con_id)
{
    // World body id = 0.
    // conbody1 && conbody2 output will be zero if either of them is worldbody
    // send yes if worldbody found
    return !(conbody1(mm, dd, con_id) && conbody2(mm, dd, con_id));
}

int nconworld(const mjModel *mm, mjData *dd, int con_id_op[])
{
    int nwcon = 0;
    for (int i = 0; i < dd->ncon; i++)
    {
        if (isconbodyworld(mm, dd, i))
        {
            con_id_op[nwcon] = i;
            nwcon++;
        }
    }
    return nwcon;
}

void contactsort(const mjModel *mm, mjData *dd, int grp_start_id_op[], int ncon_in_a_grp_op[], int *ngrp_op)
{
    // Output: Number of contact groups
    int ngrp = 0;
    // Output: Array of number of contacts in a contact group
    int ncon_in_a_grp = 0;

    int congeom1, congeom2;
    int conbody1 = -1, conbody2 = -1, conbody1old = -11, conbody2old = -11;
    for (int i = 0; i < dd->ncon; i++)
    {
        //  Find geometries in contact
        congeom1 = dd->contact[i].geom1; // dd->contact[i].geom[0]
        congeom2 = dd->contact[i].geom2; // dd->contact[i].geom[1]

        // Find bodies in contact (from geometries in contact)
        conbody1 = mm->geom_bodyid[congeom1];
        conbody2 = mm->geom_bodyid[congeom2];

        // Group the contacts according to pair of bodies
        // If either of the bodies- body1 or body2 change, assign a new group
        if ((conbody1 != conbody1old) || (conbody2 != conbody2old || i == 0))
        {
            // Output: Array of indices i where new group starts. mjdata.contact[i].
            grp_start_id_op[ngrp] = i;
            ngrp++;
        }
        conbody1old = conbody1;
        conbody2old = conbody2;
    }
    // Output: Write number of groups to the output
    *ngrp_op = ngrp;
    // printf("contact groups %d\n",ngrp);
    // congrp has indices from mjdata.contact array where a contact group starts
    // MuJoCo already has contacts sorted based on pair of geomteries

    // List number of contacts in each contact group
    for (int i = 0; i < ngrp; i++)
    {
        if (i == ngrp - 1)
        {
            ncon_in_a_grp = dd->ncon - grp_start_id_op[i];
        }
        else
        {
            ncon_in_a_grp = grp_start_id_op[i + 1] - grp_start_id_op[i];
        }
        ncon_in_a_grp_op[i] = ncon_in_a_grp;
    }
}

void contactsort_world(const mjModel *mm, mjData *dd, int wgrp_start_id_op[], int ncon_in_w_grp_op[], int *nwgrp_op)
{
    // Run contact sort first
    int grp_start_id[NCONMAX_CDA], ncon_in_a_grp[NCONMAX_CDA], ngrp;
    contactsort(mm, dd, grp_start_id, ncon_in_a_grp, &ngrp);

    // Pick those groups which have world body as one of the bodies
    int nwgrp = 0;
    for (int i = 0; i < ngrp; i++)
    {
        if (isconbodyworld(mm, dd, grp_start_id[i]))
        {
            wgrp_start_id_op[nwgrp] = grp_start_id[i];
            ncon_in_w_grp_op[nwgrp] = ncon_in_a_grp[i];
            nwgrp++;
        }
    }
    *nwgrp_op = nwgrp;
}

void contactgroupdata(const mjModel *mm, mjData *dd, int grpno, mjtNum *Rcop, mjtNum *Rcen, mjtNum *TotalContactForce)
{
    // The Rcop computations are valid only for horizontal surfaces (with normal pointing upwards)
    int congeom1, congeom2;
    int conbody1 = -1, conbody2 = -1;
    mjtNum conpos[3] = {-1., -1., -1.};
    mjtNum dimm;
    mjtNum contactFT[6] = {0., 0., 0., 0., 0., 0.};
    mjtNum contactF_XYZ[3];

    // mjtNum TotalContactForce[3] = {0., 0., 0.};
    TotalContactForce[0] = 0;
    TotalContactForce[1] = 0;
    TotalContactForce[2] = 0;
    Rcop[0] = 0.;
    Rcop[1] = 0.;
    Rcop[2] = 0.;
    Rcen[0] = 0.;
    Rcen[1] = 0.;
    Rcen[2] = 0.;
    mjtNum tempcop[3] = {0., 0., 0.}, temp[3];
    mjtNum tempinv1, tempinv2;

    int cgrp[dd->ncon];
    int ncgrp[dd->ncon];
    int ngrp;
    contactsort(mm, dd, cgrp, ncgrp, &ngrp);
    // printf("Number of contact groups: %d\n",ngrp);
    // printf("Group no needed: %d\n",grpno);
    // printf("CGRP %d\t%d\t%d\t%d\n",cgrp[0],cgrp[1],cgrp[2],cgrp[3]);
    // printf("NCon %d\t%d\t%d\t%d\n",ncgrp[0],ncgrp[1],ncgrp[2],ncgrp[3]);

    if (grpno <= ngrp - 1)
    {
        for (int i = cgrp[grpno]; i < cgrp[grpno] + ncgrp[grpno]; i++)
        {
            // Find geometries in contact
            congeom1 = dd->contact[i].geom1;
            congeom2 = dd->contact[i].geom2;

            // Position of the contact points
            mju_copy3(conpos, dd->contact[i].pos);
            // For centroid calculations
            mju_add3(Rcen, Rcen, conpos);

            // Find bodies in contact (from geometries in contact)
            conbody1 = mm->geom_bodyid[congeom1];
            conbody2 = mm->geom_bodyid[congeom2];

            if (conbody1 && conbody2)
                printf("Warning! None of the bodies is ground in this contact pair. Rcop invalid.\n");

            // printf("Contact #%d\tbodies:\t%d-%d\tGlobal coord.:\t%lg\t%lg\t%lg", i + 1, conbody1, conbody2, conpos[0], conpos[1], conpos[2]);

            // Find contact forces
            mj_contactForce(mm, dd, i, contactFT);
            // Transform the contact forces to the Global Frame of reference
            // mju_rotVecMatT(contactF_XYZ, contactFT, dd->contact[i].frame);
            mju_mulMatTVec(contactF_XYZ, dd->contact[i].frame, contactFT, 3, 3); // Get fx fy fz in the global frame
            /* It is tricky to get the ground reaction force direction correctly.
            Some times it points correctly out of the ground. But sometimes if doesn't.
            See the notes corresponding to Dec 29-30, 2022 for details.  */
            if (mm->geom_bodyid[dd->contact[i].geom1])
            {
                // flip all the grf vectors if geom1 does not belong to world body
                mju_scl3(contactF_XYZ, contactF_XYZ, -1);
            }

            // COP calculations
            // Sum of all the contact forces
            mju_add3(TotalContactForce, TotalContactForce, contactF_XYZ);
            // Fzi * Ri
            mju_scl3(temp, conpos, contactF_XYZ[2]);
            // Sum of Fzi * Ri = tempcop
            mju_add3(tempcop, tempcop, temp);
        }

        // 1/Fz
        tempinv1 = 1. / TotalContactForce[2];
        //  COP = (Sum Fzi * Ri) / Fz
        mju_scl3(Rcop, tempcop, tempinv1);

        // printf("COP coordinates:\t%lg\t%lg\n", Rcop[0], Rcop[1]);

        // Centre of Mass projection on ground
        // printf("COM projection: \t%lg\t%lg\n", dd->sensordata[com_id], dd->sensordata[com_id + 1]);

        // Geometrical Centroid of support polygon
        // This is not exactly centroid when both the feet are touching the ground
        // Edit this later during computations of feet specific COP
        tempinv2 = 1. / ncgrp[grpno];
        mju_scl3(Rcen, Rcen, tempinv2);
        // printf("Centroid coordinates:\t%lg\t%lg\n", centroid[0], centroid[1]);

        // Useful for 2.5D terrain
        Rcop[2] = Rcen[2];
    }
}

void genCOP(const mjModel *mm, mjData *dd, double g_cop_op[], ContactDataAnalysis *cda)
{
    // g_cop_op is the normal reaction weighted 3D COP of all the contacts.
    int con_ID = 0;
    double lcf[6];  // Contact force-torque at a specific contact point in the contact frame.
    double cf[3];   // Contact force-torque at a specific contact point in the global frame.
    double tnr = 0; // Total normal reaction force.
    double cpos[3]; // Point of contact.
    double gcop[3] = {0, 0, 0};
    // Iterate over contact points with the world geometries.
    for (int i = 0; i < cda->nconw; i++)
    {
        con_ID = cda->world_con_IDs[i];

        // Compute the contact force-torque in the local frame of reference.
        mj_contactForce(mm, dd, con_ID, lcf);
        mju_mulMatTVec(cf, dd->contact[con_ID].frame, lcf, 3, 3);

        // The contact force given is exerted by con.geom1 on con.geom2 according to the documentation.
        // We are interested in the force exerted by the ground/ world body.
        // Let us see if the body1 is the world body. If not, then flip the sign of the contact forces.
        if (conbody1(mm, dd, con_ID)) // This will be true if body1 is not the world body.
            mju_scl3(cf, cf, -1);     // Then flip the sign of the contact forces.

        // Compute the point of contact in the global frame of reference.
        mju_copy3(cpos, dd->contact[con_ID].pos);
        gcop[0] += cf[2] * cpos[0];
        gcop[1] += cf[2] * cpos[1];
        gcop[2] += cf[2] * cpos[2];
        tnr += cf[2];
    }

    gcop[0] /= tnr;
    gcop[1] /= tnr;
    gcop[2] /= tnr;
    mju_copy3(g_cop_op, gcop); // Write to output.
}

void footCOP(const mjModel *mm, mjData *dd, ContactDataAnalysis *cda, int wgrpID_ip, int *footBodyID_op, double footNormal_op[3], double footGRF_op[3], double cop_op[])
{
    /*
    Note1:
    Contact frame normal points from body[geom1] to body[geom2], i.e., body1 to body2.
    The corresponding contact force is exerted by body1 on body2.
    If body1 == 0 (world), then the contact force is the GRF, as expected.
    If body1 != 0, then the contact force is the opposite of the expected quantity.

    Note2:
    The foot COP is first computed in a local frame of reference.
    The local frame (lorg/Rlmat) of reference is chosen as the contact frame of the first 'valid' contact point in the group.
    The foot COP coordinates are then transformed to the global frame of reference.
    */

    // Find the body ID of the foot
    int grpStartID = cda->w_group_start_IDs[wgrpID_ip]; // Start contact ID of the group
    int LeftFoot = 0, RightFoot = 0;
    int body1 = conbody1(mm, dd, grpStartID);
    int body2 = conbody2(mm, dd, grpStartID);
    if ((body1 == LF_BODY) || (body2 == LF_BODY))
    {
        LeftFoot = 1;
        *footBodyID_op = LF_BODY;
    }
    else if ((body1 == RF_BODY) || (body2 == RF_BODY))
    {
        RightFoot = 1;
        *footBodyID_op = RF_BODY;
    }

    // Filter contacts to remove side contacts
    int nValidCon = 0;                               // Number of valid contacts to be used for calculating COP
    int valid_conID[6] = {0};                        // List of valid contacts (max 6)
    int nConInGrp = cda->ncon_in_w_group[wgrpID_ip]; // ncon in this group
    if (nConInGrp > 6)
        printf("footCOP(): too many contacts.\n");
    double valid_con_normals[3 * 6];
    // Extract only those contact normals that are vertical (<45 deg from vertical)
    double tmp_n[3], bckup, pure_z[3] = {0, 0, 1}, cs, cs_limit = 0.707; // cos(45 deg)
    for (int i = 0; i < nConInGrp; i++)
    {
        // get the contact normal (frame[0-2])
        mju_copy3(tmp_n, dd->contact[grpStartID + i].frame);

        // Check angle with {0,0,1} (vertical)
        cs = mju_dot3(tmp_n, pure_z);
        // If the normal is vertical enough,
        if (abs(cs) > cs_limit)
        {
            // Store the contact ID
            valid_conID[nValidCon] = grpStartID + i;
            // Rewrite the normal if it points downwards
            if (cs < 0) // or if(body1)
                mju_scl3(tmp_n, tmp_n, -1);
            // Store the normal
            mju_copy3(valid_con_normals + 3 * nValidCon, tmp_n);

            nValidCon++;
        }
    }

    // Within all valid contacts, check if all the contact normals are parallel
    bool single_plane_contact = true;
    double dot_prod = 0;
    for (int i = 1; i < nValidCon; i++)
    {
        // find angle w.r.t. the first normal
        dot_prod = mju_dot3(valid_con_normals, valid_con_normals + 3 * i);
        // If at least one pair of normal differs
        if (dot_prod < 0.999) // acos(0.999) = 2.56 deg
        {
            single_plane_contact = false;
            // Use foot centroid site Z axis as the common normal
            mju_copy3(valid_con_normals, dd->site_xmat + 9 * (LF_SITE * LeftFoot + RF_SITE * RightFoot) + 6);
            break;
        }
    }
    // Report the first valid normal as the foot normal
    mju_copy3(footNormal_op, valid_con_normals);

    // Compute the footGRF and the foot-COP
    if (single_plane_contact)
    {
        int con_ID = 0;
        // Define the local origin at foot site at the first contact point
        double lorg[3], lRmat[9], lRmatT[9], tmp[3];
        double lcpos[3], cpos[3], ccf[6], cf[3], lcf[3], lgcop[3] = {0., 0., 0.}, tnr = 0., grf[3] = {0., 0., 0.};
        for (int i = 0; i < nConInGrp; i++)
        {
            con_ID = valid_conID[i];

            // Compute the local origin and the local frame of reference
            if (i == 0)
            {
                mju_copy3(lorg, dd->contact[con_ID].pos);
                mju_copy(lRmatT, dd->contact[con_ID].frame, 9);
                mju_transpose(lRmat, lRmatT, 3, 3);
            }

            // Point of contact in the global frame of reference
            mju_copy3(cpos, dd->contact[con_ID].pos);

            // Point of contact w.r.t the local origin (global frame)
            mju_sub3(tmp, cpos, lorg);
            // global to local
            mju_mulMatTVec(lcpos, lRmat, tmp, 3, 3);

            // Contact force-torque in the contact frame
            mj_contactForce(mm, dd, con_ID, ccf);
            // Contact force in the global frame
            mju_mulMatTVec(cf, dd->contact[con_ID].frame, ccf, 3, 3);
            // Correct the contact force if it is the opposite of what is expected
            if (cf[2] < 0)
                mju_scl3(cf, cf, -1);

            // Contact force in the local frame
            mju_mulMatTVec(lcf, lRmat, cf, 3, 3);

            // lcf[0] is the normal force in the local frame.
            lgcop[0] += lcf[0] * lcpos[0];
            lgcop[1] += lcf[0] * lcpos[1];
            lgcop[2] += lcf[0] * lcpos[2];
            tnr += lcf[0];

            // foot GRF
            grf[0] += cf[0];
            grf[1] += cf[1];
            grf[2] += cf[2];
        }

        lgcop[0] /= tnr;
        lgcop[1] /= tnr;
        lgcop[2] /= tnr;

        // Transform the foot COP to the global frame of reference
        // printf("lgcop: %lg\t%lg\t%lg\n", lgcop[0], lgcop[1], lgcop[2]);
        mju_mulMatVec(tmp, lRmat, lgcop, 3, 3);
        mju_add3(cop_op, tmp, lorg);
        mju_copy3(footGRF_op, grf);
    }
    else
    {
        // Work out the logic.
        printf("footCOP(): Multiple planes of contact for single foot. Not implemented yet.\n");
    }
}

void genCOP1(const mjModel *mm, mjData *dd, ContactDataAnalysis *cda, double lf_cop_op[], double rf_cop_op[], double g_cop_op[])
{
    // get the number of contact groups
    int nwcongrps = cda->nwgroups;
    if (cda->nconw)
    {
        // Proceed only if the robot is in contact with the world
        if (nwcongrps > 2) // robot declared as fallen
        {
            mju_zero3(lf_cop_op);
            mju_zero3(rf_cop_op);
            mju_zero3(g_cop_op);
        }
        else if (nwcongrps == 1) // Single leg stance
        {
            // Foot COP is the 3D COP-ZMP
            int whichFoot;
            // fN is contact normal, always points upwards, cf is contact force in the global frame, always points upwards
            double fN[3], cf[3];
            footCOP(mm, dd, cda, 0, &whichFoot, fN, cf, g_cop_op);
            if (whichFoot == LF_BODY)
            {
                mju_copy3(lf_cop_op, g_cop_op);
                mju_zero3(rf_cop_op);
            }
            else
            {
                mju_copy3(rf_cop_op, g_cop_op);
                mju_zero3(lf_cop_op);
            }
        }
        else // Double leg stance
        {
            int whichFoot1, whichFoot2;
            // Foot COP data of one foot
            double n1[3], f1[3], g_cop1[3];
            // Contact group 0
            footCOP(mm, dd, cda, 0, &whichFoot1, n1, f1, g_cop1);
            // debug print f1 and gcop1
            // printf("f1: %lg\t%lg\t%lg\n", f1[0], f1[1], f1[2]);
            // printf("g_cop1: %lg\t%lg\t%lg\n", g_cop1[0], g_cop1[1], g_cop1[2]);

            // Foot COP data of the other foot
            double n2[3], f2[3], g_cop2[3];
            // Contact group 1
            footCOP(mm, dd, cda, 1, &whichFoot2, n2, f2, g_cop2);
            // debug print f2 and gcop2
            // printf("f2: %lg\t%lg\t%lg\n", f2[0], f2[1], f2[2]);
            // printf("g_cop2: %lg\t%lg\t%lg\n", g_cop2[0], g_cop2[1], g_cop2[2]);

            // Copy foot COP data to the output
            if (whichFoot1 == LF_BODY)
            {
                mju_copy3(lf_cop_op, g_cop1);
                mju_copy3(rf_cop_op, g_cop2);
            }
            else
            {
                mju_copy3(lf_cop_op, g_cop2);
                mju_copy3(rf_cop_op, g_cop1);
            }

            // Check if the feet are parallel
            double parallel = mju_dot3(n1, n2);
            double tol_parallel = 0.9997;
            if (parallel > tol_parallel)
            {
                // Both the feet are parallel: flat ground or steps
                double Navg[3];
                mju_add3(Navg, n1, n2); // no need actually Navg = n1 = n2
                mju_normalize3(Navg);
                // Compute the genCOP in the world frame as the weighted average of the two foot COPs (each in the world frame)
                double fz1 = f1[2], fz2 = f2[2];
                g_cop_op[0] = (g_cop1[0] * fz1 + g_cop2[0] * fz2) / (fz1 + fz2);
                g_cop_op[1] = (g_cop1[1] * fz1 + g_cop2[1] * fz2) / (fz1 + fz2);
                g_cop_op[2] = (g_cop1[2] * fz1 + g_cop2[2] * fz2) / (fz1 + fz2);

                // printf("g_cop_op: %lg\t%lg\t%lg\n", g_cop_op[0], g_cop_op[1], g_cop_op[2]);
            }
            else
            {
                // The feet are not parallel
                double u[3], Navg[3];
                mju_cross(u, n1, n2);
                mju_normalize3(u);

                // Normal to the average plane
                double fz1 = f1[2], fz2 = f2[2];
                Navg[0] = fz1 * n1[0] + fz2 * n2[0];
                Navg[1] = fz1 * n1[1] + fz2 * n2[1];
                Navg[2] = fz1 * n1[2] + fz2 * n2[2];
                mju_normalize3(Navg);

                // Compute a point on the average plane (call it o1)
                double t1[3];
                mju_cross(t1, n1, u);
                mju_normalize3(t1);

                double c1mc2[3];
                mju_sub3(c1mc2, g_cop1, g_cop2);
                // Distance from the one foot COP to the foot of perpendicular on the common line between the planes
                double d1 = -mju_dot3(c1mc2, n2) / mju_dot3(t1, n2);
                mju_scl3(t1, t1, d1);
                double o1[3];
                mju_add3(o1, g_cop1, t1);

                // Compute the gCOP on the average plane using (o1, Navg)
                double ro1c1[3], ro1c2[3];
                mju_sub3(ro1c1, g_cop1, o1);
                mju_sub3(ro1c2, g_cop2, o1);
                double m1_o1[3], m2_o1[3], m_o1[3];
                mju_cross(m1_o1, ro1c1, f1);
                mju_cross(m2_o1, ro1c2, f2);
                mju_add3(m_o1, m1_o1, m2_o1);
                double f1pf2[3];
                mju_add3(f1pf2, f1, f2);
                double num[3], den;
                mju_cross(num, Navg, m_o1);
                den = mju_dot3(Navg, f1pf2);
                den = 1 / den;
                double r_o1_c[3];
                mju_scl3(r_o1_c, num, den);

                // Get the coordinates of the gCOP in the world frame.
                mju_add3(g_cop_op, o1, r_o1_c);
                // printf("g_cop_op: %lg\t%lg\t%lg\n", g_cop_op[0], g_cop_op[1], g_cop_op[2]);
            }
        }
    }
    else
    {
        // No contact with the world
        mju_zero3(lf_cop_op);
        mju_zero3(rf_cop_op);
        mju_zero3(g_cop_op);
    }
    // mju_printMat(lf_cop_op, 1, 3);
    // mju_printMat(rf_cop_op, 1, 3);
    // mju_printMat(g_cop_op, 1, 3);
}

void printcontactdata(mjModel *mm, mjData *dd)
{

    mjtNum conpos[3] = {-1., -1., -1.};
    int congeom1, congeom2, conbody1, conbody2;
    mjtNum dimm;
    mjtNum contactFT[6] = {0., 0., 0., 0., 0., 0.};
    mjtNum contactF_XYZ[3];

    mjtNum TotalContactForce[3] = {0., 0., 0.};
    mjtNum Rcop[3], centroid[3] = {0., 0., 0.};
    mjtNum tempcop[3] = {0., 0., 0.}, temp[3];
    mjtNum tempinv1, tempinv2;
    int com_id = 0;

    for (int i = 0; i < dd->ncon; i++)
    {
        // Find geometries in contact
        congeom1 = dd->contact[i].geom1;
        congeom2 = dd->contact[i].geom2;
        // Position of the contact points
        mju_copy3(conpos, dd->contact[i].pos);
        // For centroid calculations
        mju_add3(centroid, centroid, conpos);
        // Find bodies in contact (from geometries in contact)
        conbody1 = mm->geom_bodyid[congeom1];
        conbody2 = mm->geom_bodyid[congeom2];
        printf("Contact #%d\tbodies:\t%d-%d\tGlobal coord.:\t%lg\t%lg\t%lg", i + 1, conbody1, conbody2, conpos[0], conpos[1], conpos[2]);

        // Find contact forces
        mj_contactForce(mm, dd, i, contactFT);
        // Transform the contact forces to the Global Frame of reference
        // mju_rotVecMatT(contactF_XYZ, contactFT, dd->contact[i].frame); // Deprecated since MuJoCo 3.2.0
        mju_mulMatTVec3(contactF_XYZ, dd->contact[i].frame, contactFT); // Get fx fy fz in the global frame
        printf("\tContact force:\t%lg\t%lg\t%lg\n", contactF_XYZ[0], contactF_XYZ[1], contactF_XYZ[2]);

        // COP calculations
        // Sum of all the contact forces
        mju_add3(TotalContactForce, TotalContactForce, contactF_XYZ);
        // Fzi * Ri
        mju_scl3(temp, conpos, contactF_XYZ[2]);
        // Sum of Fzi * Ri
        mju_add3(tempcop, tempcop, temp);
    }
    // COP = (Sum Fzi * Ri) / Fz
    // Ignore the Z component of Rcop
    tempinv1 = 1. / TotalContactForce[2];
    mju_scl3(Rcop, tempcop, tempinv1);
    printf("COP coordinates:\t%lg\t%lg\n", Rcop[0], Rcop[1]);

    // Centre of Mass projection on ground
    printf("COM projection: \t%lg\t%lg\n", dd->sensordata[com_id], dd->sensordata[com_id + 1]);

    // Geometrical Centroid of support polygon
    // This is not exactly centroid when both the feet are touching the ground
    // Edit this later during computations of feet specific COP
    tempinv2 = 1. / dd->ncon;
    mju_scl3(centroid, centroid, tempinv2);
    printf("Centroid coordinates:\t%lg\t%lg\n", centroid[0], centroid[1]);
}

double mapValue(double v, double vmin, double vmax, double op_min, double op_max)
{
    return op_min + (op_max - op_min) * ((v - vmin) / (vmax - vmin));
}

void proj_contact_points(double focal_point[], double ip_points[], int n_points, double op_z, double op_points[])
{
    double fx = focal_point[0];
    double fy = focal_point[1];
    double fz = focal_point[2];

    for (int i = 0; i < n_points; i++)
    {
        op_points[3 * i] = fx + (ip_points[3 * i] - fx) * (op_z - fz) / (ip_points[3 * i + 2] - fz);         // X coordinate
        op_points[3 * i + 1] = fy + (ip_points[3 * i + 1] - fy) * (op_z - fz) / (ip_points[3 * i + 2] - fz); // Y coordinate
        op_points[3 * i + 2] = op_z;                                                                         // Z coordinate
    }
}

double mydist(const double p1[], const double p2[], const int dim)
{
    double op = 0;
    for (int i = 0; i < dim; i++)
    {
        op += (p1[i] - p2[i]) * (p1[i] - p2[i]);
    }

    return sqrt(op);
}

void farthest_points(const double polygon[], const int dim, const int nvert, int *op_p1_id, int *op_p2_id)
{
    double d, maxDist = 0;
    // O(n^2) algo
    for (int i = 0; i < nvert; i++)
    {
        for (int j = i + 1; j < nvert; j++)
        {
            d = mydist(&polygon[i * dim], &polygon[j * dim], dim);
            if (d > maxDist)
            {
                maxDist = d;
                *op_p1_id = i;
                *op_p2_id = j;
            }
        }
    }
    // Expand functionality if needed, find second farthest pair and so on.
}

void farthest_points_axis(const double polygon[], const int dim, const int nvert, double op_t_axis[])
{
    // Take a polygon as input (mostly from the output of contact_cHull)
    // Find the farthest points
    int id1, id2;
    double p1[3], p2[3];
    farthest_points(polygon, dim, nvert, &id1, &id2);
    mju_copy3(p1, polygon + dim * id1);
    mju_copy3(p2, polygon + dim * id2);
    // Find the vector difference
    mju_sub3(op_t_axis, p1, p2);
    printf("Tipping axis in farthest_points_axis: %lg\t%lg\t%lg\n", op_t_axis[0], op_t_axis[1], op_t_axis[2]);
    // Normalize and send out the tipping axis
    mju_normalize3(op_t_axis);
}

void cHull2D(double ip_multi_point2D[], int ip_len, double op_poly2D[], int *op_poly2D_len, double *spa)
{
    namespace bg = boost::geometry;
    namespace bgm = boost::geometry::model;

    // Initializations of point polygon template
    typedef bgm::point<double, 2, bg::cs::cartesian> point_t;
    typedef bg::model::multi_point<point_t> multipoint_t;
    typedef bg::model::polygon<point_t> polygon_t;

    multipoint_t ip_pts;
    polygon_t c_hull;

    // Initialize the multi_point object from a double array
    for (int i = 0; i < ip_len; i++)
    {
        bg::append(ip_pts, point_t(ip_multi_point2D[2 * i], ip_multi_point2D[2 * i + 1]));
    }

    // Find convex hull polygon
    bg::convex_hull(ip_pts, c_hull);
    // Find area of the convex polygon
    double c_hull_area = bg::area(c_hull);
    // Find perimeter of the convex polygon
    // double c_hull_perimeter = bg::perimeter(c_hull);

    // if (c_hull_perimeter)
    // {
    //     *sk = c_hull_area / (c_hull_perimeter * c_hull_perimeter); // Skewness a/p^2
    // }
    // else
    //     *sk = 0; //(sk = 0 if p == 0)

    *spa = c_hull_area;

    // Extract the points from the convex polygon to the output C array
    int i = 0;
    // printf("Convex hull pts:\n");
    for (auto &point : c_hull.outer())
    {
        op_poly2D[2 * i] = bg::get<0>(point);
        op_poly2D[2 * i + 1] = bg::get<1>(point);
        // printf("(%lg,%lg)\n", op_poly2D[2 * i], op_poly2D[2 * i + 1]);
        i++;
    }
    *op_poly2D_len = i;
    // printf("c hull poly length %d\n", *op_poly2D_len);
}

void contact_cHull(const mjModel *mm, mjData *dd, double op_poly2D[], int *op_poly2D_len, double *s_p_a)
{
    // No safety check for d->ncon
    int world_con_id[dd->ncon];
    // nconw <= dd->ncon
    int nconw = nconworld(mm, dd, world_con_id);
    // Use only contacts with the ground (worldbody)
    double world_con_xy[2 * nconw];

    // Get X and Y coordinates of the contacts with the world body
    for (int i = 0; i < nconw; i++)
    {
        world_con_xy[2 * i] = dd->contact[world_con_id[i]].pos[0];     // X coord
        world_con_xy[2 * i + 1] = dd->contact[world_con_id[i]].pos[1]; // Y coord
    }

    cHull2D(world_con_xy, dd->ncon, op_poly2D, op_poly2D_len, s_p_a);
    // mju_printMat(op_poly2D, 1, *op_poly2D_len * 2);
    // printf("Support Polygon Area SPA: %lg\n",*skn);
}

bool check_interior(const double gcz[], const double el_cx, const double el_cy, const double el_a, const double el_b)
{
    

    // Check if the ZMP lies inside the ellipse
    double x = gcz[0] - el_cx;
    double y = gcz[1] - el_cy;
    bool op = false;
    return op = (x * x) / (el_a * el_a) + (y * y) / (el_b * el_b) <= 1;
}

void my_mju_relQuat(mjtNum *qdif, const mjtNum qa[4], const mjtNum qb[4])
{
    // qdif = neg(qb)*qa
    mjtNum qneg[4];
    mju_negQuat(qneg, qb);
    mju_mulQuat(qdif, qneg, qa);
}

void my_mj_differentiateXPos(mjtNum *qvel, mjtNum dt, const mjtNum *qpos1, const mjtNum *qpos2)
{
    mjtNum neg[4], dif[4];
    // X Y Z
    for (int i = 0; i < 3; i++)
    {
        qvel[i] = (qpos2[i] - qpos1[i]) / dt;
    }

    // rotations
    mju_negQuat(neg, qpos1 + 3); // solve:  qpos1 * dif = qpos2
    mju_mulQuat(dif, neg, qpos2 + 3);
    mju_quat2Vel(qvel + 3, dif, dt);

    // Subtract quaternions, express as 3D velocity: qb*quat(res) = qa.
    /* void mju_subQuat(mjtNum res[3], const mjtNum qa[4], const mjtNum qb[4]) {
      // qdif = neg(qb)*qa
      mjtNum qneg[4], qdif[4];
      mju_negQuat(qneg, qb);
      mju_mulQuat(qdif, qneg, qa);

      // convert to 3D velocity
      mju_quat2Vel(res, qdif, 1);
    } */
}

void my_mju_quat2axisAngle(const mjtNum *quat, mjtNum *axisa)
{
    double iquat[4]; // internal quaternion
    mju_copy4(iquat, quat);
    // mju_normalize4(iquat);
    double q0 = iquat[0], q1 = iquat[1], q2 = iquat[2], q3 = iquat[3];
    double temp, sinphiby2;

    // Choose the positive angle. There are two possible axis-angle pairs for a given quaternion.
    sinphiby2 = mju_sqrt(q1 * q1 + q2 * q2 + q3 * q3);
    
    // Once the angle sign of sinphiby2 is fixed, the angle is determined uniquely using atan2
    axisa[3] = atan2(sinphiby2, q0) * 2;

    // When q0 == + or -1
    if (abs(q0 * q0 - 1) < 1e-6)
    {
        axisa[0] = 0;
        axisa[1] = 0;
        axisa[2] = 1;
    }
    else
    {
        temp = mju_sqrt(1 - q0 * q0);
        axisa[0] = q1 / temp;
        axisa[1] = q2 / temp;
        axisa[2] = q3 / temp;
    }
}

void my_mju_mat2axisAngle(const mjtNum *mat, mjtNum *axisa)
{
    // Write the logic from scratch if needed
    double q[4];
    mju_mat2Quat(q, mat);
    my_mju_quat2axisAngle(q, axisa);
}

void my_mju_avgRmat(const mjtNum *Rmat1, const mjtNum *Rmat2, mjtNum *op_Rmat)
{
    mjtNum R_diff[9], quat_diff[4], axisa_diff[4];
    mju_mulMatTMat(R_diff, Rmat1, Rmat2, 3, 3, 3);
    mju_mat2Quat(quat_diff, R_diff);
    my_mju_quat2axisAngle(quat_diff, axisa_diff);
    axisa_diff[3] = 0.5 * axisa_diff[3];
    mju_axisAngle2Quat(quat_diff, axisa_diff, axisa_diff[3]);
    mju_quat2Mat(R_diff, quat_diff);
    mju_mulMatMatT(op_Rmat, Rmat1, R_diff, 3, 3, 3);
}

void my_mju_kphi_equiv(double incr_k_phi[], double curr_k_phi[], double op_k_phi[])
{
    // through quaternions
    double q_curr[4], q_incr[4], q_op[4];
    mju_axisAngle2Quat(q_curr, curr_k_phi, curr_k_phi[3]);
    mju_axisAngle2Quat(q_incr, incr_k_phi, incr_k_phi[3]);
    mju_mulQuat(q_op, q_incr, q_curr);
    my_mju_quat2axisAngle(q_op, op_k_phi);

    // defining direction of new k
    // if (mju_sign(curr_k_phi[3]) != mju_sign(op_k_phi[3]))
    // {
    //     mju_scl(op_k_phi, op_k_phi, -1, 4); // Flip to -k -phi
    // }
}

void bioloid_12dof_IK_position(const mjModel *mm, int tsno, int lfsno, int rfsno,
                               double global_torso[], double global_torso_kphi[],
                               double global_leftfoot[], double global_leftfoot_kphi[],
                               double global_rightfoot[], double global_rightfoot_kphi[],
                               double qj_op[],
                               int nitrmax, double ef, double ex)
{
    bool llsolconv = true, rlsolconv = true;
    /*---------Processing Inputs---Convert axis angle to quat------------------------*/
    double global_torso_quat[4], global_leftfoot_quat[4], global_rightfoot_quat[4];
    double global_torso_kaxis[3] = {global_torso_kphi[0], global_torso_kphi[1], global_torso_kphi[2]};
    double global_leftfoot_kaxis[3] = {global_leftfoot_kphi[0], global_leftfoot_kphi[1], global_leftfoot_kphi[2]};
    double global_rightfoot_kaxis[3] = {global_rightfoot_kphi[0], global_rightfoot_kphi[1], global_rightfoot_kphi[2]};
    mju_normalize3(global_torso_kaxis);
    mju_normalize3(global_leftfoot_kaxis);
    mju_normalize3(global_rightfoot_kaxis);
    mju_axisAngle2Quat(global_torso_quat, global_torso_kaxis, global_torso_kphi[3]);
    mju_axisAngle2Quat(global_leftfoot_quat, global_leftfoot_kaxis, global_leftfoot_kphi[3]);
    mju_axisAngle2Quat(global_rightfoot_quat, global_rightfoot_kaxis, global_rightfoot_kphi[3]);

    /*--make mjData instance for IK problem Jacobian computation----------------------*/
    mjData *ik_d = mj_makeData(mm);

    // Position of the site w.r.t the floating base
    double torso_site_offsetpos[3], torso_site_offsetquat[4];
    mju_copy3(torso_site_offsetpos, mm->site_pos + tsno * 3);   // Correct
    mju_copy4(torso_site_offsetquat, mm->site_quat + tsno * 4); // Correct

    // Solve for the floating base orientation give the torso site orientation
    my_mju_relQuat(ik_d->qpos + 3, global_torso_quat, torso_site_offsetquat); // Correct
    // Solve for the floating base position given the torso site position and orientation
    double rot_fb[9];
    mju_quat2Mat(rot_fb, ik_d->qpos + 3);
    mju_mulMatVec(ik_d->qpos, rot_fb, torso_site_offsetpos, 3, 3);
    mju_sub3(ik_d->qpos, global_torso, ik_d->qpos);
    //  Rest of the joints qj = [qjL|qjR]
    mju_copy(ik_d->qpos + 7, qj_op, 12);

    // Turn off collision detection for this mjData??
    mj_step1(mm, ik_d);

    /*     printf("IK function q\n");
        for (int i = 0; i < mm->nq; i++)
            printf("%.4f\t", ik_d->qpos[i]);
        printf("\n\n"); */

    /*--Compute the linear velocity and angular velocity Jacobian matrices-------------*/
    double ll_jacV[3 * 18], ll_jacW[3 * 18], rl_jacV[3 * 18], rl_jacW[3 * 18];
    mj_jacSite(mm, ik_d, ll_jacV, ll_jacW, lfsno);
    mj_jacSite(mm, ik_d, rl_jacV, rl_jacW, rfsno);
    // Entry--------Eigen------->> whistle podu
    Eigen::Map<Eigen::Matrix<double, 3, 18, Eigen::RowMajor == 1>> LL_JacV(ll_jacV, 3, 18);
    Eigen::Map<Eigen::Matrix<double, 3, 18, Eigen::RowMajor == 1>> LL_JacW(ll_jacW, 3, 18);
    Eigen::Map<Eigen::Matrix<double, 3, 18, Eigen::RowMajor == 1>> RL_JacV(rl_jacV, 3, 18);
    Eigen::Map<Eigen::Matrix<double, 3, 18, Eigen::RowMajor == 1>> RL_JacW(rl_jacW, 3, 18);
    // Jjl is a 6x6 matrix mapping qjl_dot to [Vlx Vly Vlz Wlx Wly Wlz]
    // Jjr is a 6x6 matrix mapping qjr_dot to [Vrx Vry Vrz Wrx Wry Wrz]
    Eigen::Matrix<double, 6, 6, Eigen::RowMajor == 1> Jjl;
    Jjl.block<3, 6>(0, 0) = LL_JacV.block<3, 6>(0, 6);
    Jjl.block<3, 6>(3, 0) = LL_JacW.block<3, 6>(0, 6);
    Eigen::Matrix<double, 6, 6, Eigen::RowMajor == 1> Jjr;
    Jjr.block<3, 6>(0, 0) = RL_JacV.block<3, 6>(0, 12);
    Jjr.block<3, 6>(3, 0) = RL_JacW.block<3, 6>(0, 12);

    /*---Solving the inverse kinematics problem for the left leg------------------*/
    int itr = 0;
    double errorf = 1000, errorX = 1000;
    double ll_residual[6], qjl_op[6];
    Eigen::Map<Eigen::VectorXd> eig_ll_residual(ll_residual, 6);
    mju_copy(qjl_op, qj_op, 6);
    Eigen::Map<Eigen::VectorXd> eig_qjl_op(qjl_op, 6);
    Eigen::VectorXd dql_nm(6); // dql_nm: dq generated for the left leg using Newton's Method

    double left_foot_site_curr[7], left_foot_site_target[7];    // Left foot site pose: [x y z | e0 e1 e2 e3]
    mju_copy3(left_foot_site_target, global_leftfoot);          // built from function inputs
    mju_copy4(left_foot_site_target + 3, global_leftfoot_quat); // built from function inputs

    // mj_step1 has been computed
    while (errorf > ef && errorX > ex && itr < nitrmax)
    {
        // compute left foot site frame current pos // 0-1-2 3-4-5-6 are torso site
        mju_copy(left_foot_site_curr, ik_d->sensordata + 7, 7); // 7-8-9 10-11-12-13 are left foot
        // compute residual6x1 = Current frame pose [x y z | e0 e1 e2 e3] - Target frame pose
        mju_sub3(ll_residual, left_foot_site_curr, left_foot_site_target);
        mju_subQuat(ll_residual + 3, left_foot_site_curr + 3, left_foot_site_target + 3);

        /*         mju_printMat(left_foot_site_curr, 1, 7);
                mju_printMat(left_foot_site_target, 1, 7);
                mju_printMat(residual, 1, 6); */
        // std::cout << "residual:" << eig_residual.transpose() << std::endl;

        // errorf is norm(residual)
        // This is problematic because of three linear velocities and three angular velocities
        errorf = eig_ll_residual.norm();

        // dq_nm = -Jjl.inverse() * eig_residual;
        dql_nm = Jjl.fullPivHouseholderQr().solve(-eig_ll_residual);
        eig_qjl_op += dql_nm;

        errorX = dql_nm.norm();

        // Write to ik_d data
        mju_copy(ik_d->qpos + 7, qjl_op, 6);
        // mj_step1() with the updated solution of qjl
        mj_step1(mm, ik_d);
        // Compute Jacobian with updated qjl
        mj_jacSite(mm, ik_d, ll_jacV, ll_jacW, 1); // Left foot site ID = 1

        // Update the mapped Eigen matrices: LL_JacV, LL_JacW-----No need. They get updated automatically
        // Update Jjl
        Jjl.block<3, 6>(0, 0) = LL_JacV.block<3, 6>(0, 6);
        Jjl.block<3, 6>(3, 0) = LL_JacW.block<3, 6>(0, 6);

        // printf("Left foot IK: errorf: %lg \t errorQ: %lg \t iterations: %d\n", errorf, errorX, itr);
        //   mju_printMat(ik_d->qpos,19,1);

        itr++;
    }

    // Convergence notification
    if (errorf > ef && errorX > ex)
    {
        llsolconv = false;
        printf("LL sol not converged. Errorf: %lg \t ErrorQ: %lg \t Nitr: %d\n", errorf, errorX, itr);
    }
    else
        mju_copy(qj_op, qjl_op, 6);

    /*---Solving the inverse kinematics problem for the right leg------------------*/
    errorf = 1000;
    errorX = 1000;
    itr = 0;
    double rl_residual[6], qjr_op[6];
    Eigen::Map<Eigen::VectorXd> eig_rl_residual(rl_residual, 6);
    mju_copy(qjr_op, qj_op + 6, 6); // last 6 elements are the right leg joint angles
    Eigen::Map<Eigen::VectorXd> eig_qjr_op(qjr_op, 6);
    Eigen::VectorXd dqr_nm(6); // dqr_nm: dq generated for the right leg using Newton's Method

    double right_foot_site_curr[7], right_foot_site_target[7];    // Right foot site pose: [x y z | e0 e1 e2 e3]
    mju_copy3(right_foot_site_target, global_rightfoot);          // built from function inputs
    mju_copy4(right_foot_site_target + 3, global_rightfoot_quat); // built from function inputs

    // mj_step1 has been computed
    // printf("\n");
    while (errorf > ef && errorX > ex && itr < nitrmax)
    {
        // compute right foot site frame current pos // 0-1-2 3-4-5-6 are torso site
        mju_copy(right_foot_site_curr, ik_d->sensordata + 7 + 7, 7); // 14-15-16 17-18-19-20 are right foot
        // compute residual6x1 = Current frame pose [x y z | e0 e1 e2 e3] - Target frame pose
        mju_sub3(rl_residual, right_foot_site_curr, right_foot_site_target);
        mju_subQuat(rl_residual + 3, right_foot_site_curr + 3, right_foot_site_target + 3);

        /*         mju_printMat(right_foot_site_curr, 1, 7);
                mju_printMat(right_foot_site_target, 1, 7);
                mju_printMat(rl_residual, 1, 6); */
        // std::cout << "residual:" << eig_rl_residual.transpose() << std::endl;

        // errorf is norm(residual)
        // This is problematic because of three linear velocities and three angular velocities
        errorf = eig_rl_residual.norm();

        // dq_nm = -Jjr.inverse() * eig_rl_esidue;
        dqr_nm = Jjr.fullPivHouseholderQr().solve(-eig_rl_residual);
        eig_qjr_op += dqr_nm;

        errorX = dqr_nm.norm();

        // Write to ik_d data
        mju_copy(ik_d->qpos + 7 + 6, qjr_op, 6);
        // mj_step1() with the updated solution of qjr
        mj_step1(mm, ik_d);
        // Compute Jacobian with updated qjr
        mj_jacSite(mm, ik_d, rl_jacV, rl_jacW, 2); // Right foot site ID = 2

        // Update the mapped Eigen matrices: RL_JacV, RL_JacW-----No need. They get updated automatically
        // Update Jjr
        Jjr.block<3, 6>(0, 0) = RL_JacV.block<3, 6>(0, 12);
        Jjr.block<3, 6>(3, 0) = RL_JacW.block<3, 6>(0, 12);

        // printf("Right leg IK: errorf: %lg \t errorQ: %lg \t iterations: %d\n", errorf, errorX, itr);
        //  mju_printMat(ik_d->qpos,19,1);

        itr++;
    }

    // Convergence notification
    if (errorf > ef && errorX > ex)
    {
        rlsolconv = false;
        printf("RL sol not converged. Errorf: %lg \t ErrorQ: %lg \t Nitr: %d\n", errorf, errorX, itr);
    }
    else
        mju_copy(qj_op + 6, qjr_op, 6);

    // Get the angles back in domain -pi to pi
    // May be problematic?
    for (int i = 0; i < 12; i++)
        qj_op[i] = wrap_angles(qj_op[i]);

    mj_deleteData(ik_d);
    // Debugging:
    /*     printf("\n Debugging q\n");
        for (int i = 0; i < 12; i++)
            printf("%.4f,", qj_op[i]);
        printf("\n");
        printf("\n Debugging q\n");
        for (int i = 0; i < 12; i++)
            printf("%.4f\t", qj_op[i]);
        printf("\n\n"); */
}

void bioloid_12dof_IK_position_v2(const mjModel *mm, int tsno, int lfsno, int rfsno,
                                  double global_torso[], double global_torso_kphi[],
                                  double global_leftfoot[], double global_leftfoot_kphi[],
                                  double global_rightfoot[], double global_rightfoot_kphi[],
                                  double qj_op[],
                                  int nitrmax, double ef, double ex)
{
    bool solconv = true;
    /*---------Processing Inputs---Convert axis angle to quat------------------------*/
    double global_torso_kaxis[3] = {global_torso_kphi[0], global_torso_kphi[1], global_torso_kphi[2]};
    double global_leftfoot_kaxis[3] = {global_leftfoot_kphi[0], global_leftfoot_kphi[1], global_leftfoot_kphi[2]};
    double global_rightfoot_kaxis[3] = {global_rightfoot_kphi[0], global_rightfoot_kphi[1], global_rightfoot_kphi[2]};
    mju_normalize3(global_torso_kaxis);
    mju_normalize3(global_leftfoot_kaxis);
    mju_normalize3(global_rightfoot_kaxis);
    double global_torso_quat[4], global_leftfoot_quat[4], global_rightfoot_quat[4];
    mju_axisAngle2Quat(global_torso_quat, global_torso_kaxis, global_torso_kphi[3]);
    mju_axisAngle2Quat(global_leftfoot_quat, global_leftfoot_kaxis, global_leftfoot_kphi[3]);
    mju_axisAngle2Quat(global_rightfoot_quat, global_rightfoot_kaxis, global_rightfoot_kphi[3]);

    /*--make mjData instance for IK problem Jacobian computation----------------------*/
    mjData *ik_d = mj_makeData(mm);
    double z3[3] = {0., 0., 0.}, q_eye4[4] = {1, 0, 0, 0}, internal_qj_op[12];
    // Now the floating base is at origin with default orientation--->
    mju_copy3(ik_d->qpos, z3);
    mju_copy4(ik_d->qpos + 3, q_eye4);
    /*
    All calculations in Torso site frame. It has a constant offset w.r.t. the floating base
    left foot pos and orientation,
    right foot and orientation.
    */

    // Create copy of input estimate and update it. Copy back the answer at last if the iterations converge
    mju_copy(internal_qj_op, qj_op, 12);
    Eigen::Map<Eigen::VectorXd> eig_qj_op(internal_qj_op, 12); // Map Eigen vector to internal_qj_op array

    double target_lt_pos[3], target_rt_pos[3], T_target_lt_pos[3], T_target_rt_pos[3];
    double target_lt_quat[4], target_rt_quat[4];
    double R0T[9], RT0[9];
    mju_quat2Mat(R0T, global_torso_quat);
    mju_transpose(RT0, R0T, 3, 3);
    my_mju_relQuat(target_lt_quat, global_leftfoot_quat, global_torso_quat);  // Set desired orientation
    my_mju_relQuat(target_rt_quat, global_rightfoot_quat, global_torso_quat); // Set desied orientation
    mju_sub3(target_lt_pos, global_leftfoot, global_torso);
    mju_mulMatVec(T_target_lt_pos, RT0, target_lt_pos, 3, 3); // Set desired position in Torso frame: RT0(Xl-Xt)
    mju_sub3(target_rt_pos, global_rightfoot, global_torso);
    mju_mulMatVec(T_target_rt_pos, RT0, target_rt_pos, 3, 3); // Set desired position in Torso frame: RT0(Xr-Xt)

    /*--Compute the linear velocity and angular velocity Jacobian matrices-------------*/
    double l_jacV[3 * 18], l_jacW[3 * 18], r_jacV[3 * 18], r_jacW[3 * 18];

    Eigen::Map<Eigen::Matrix<double, 3, 18, Eigen::RowMajor == 1>> L_JacV(l_jacV, 3, 18);
    Eigen::Map<Eigen::Matrix<double, 3, 18, Eigen::RowMajor == 1>> L_JacW(l_jacW, 3, 18);
    Eigen::Map<Eigen::Matrix<double, 3, 18, Eigen::RowMajor == 1>> R_JacV(r_jacV, 3, 18);
    Eigen::Map<Eigen::Matrix<double, 3, 18, Eigen::RowMajor == 1>> R_JacW(r_jacW, 3, 18);
    Eigen::Matrix<double, 12, 12, Eigen::RowMajor == 1> J; // Matrix gradient of residual(qj)

    double residual[12]; // Equation: residual(qj) = 0 is being solved
    Eigen::Map<Eigen::VectorXd> eig_residual(residual, 12);

    double T_curr_lt_pos[3], T_curr_rt_pos[3], curr_lt_quat[4], curr_rt_quat[4];
    Eigen::VectorXd dqj_nm(12); // Output of Newton's method

    int itr = 0;
    double errorf = 1000, errorX = 1000;
    while (errorf > ef && errorX > ex && itr < nitrmax)
    {
        // Set the new qj for revised computations
        mju_copy(ik_d->qpos + 7, internal_qj_op, 12);
        // Compute mj_step1()
        mj_step1(mm, ik_d);
        // Compute Jacobian matrices in Torso frame
        mj_jacSite(mm, ik_d, l_jacV, l_jacW, lfsno);
        mj_jacSite(mm, ik_d, r_jacV, r_jacW, rfsno);

        // Read current position and orientation values (These values are in torso frame only becuase ik_d is set to torso frame)
        my_mju_relQuat(curr_lt_quat, ik_d->sensordata + 10, ik_d->sensordata + 3); // Orientation of left foot w.r.t. torso (Orientation of torso is not going to change!)
        my_mju_relQuat(curr_rt_quat, ik_d->sensordata + 17, ik_d->sensordata + 3); // Orientation of right foot w.r.t. torso (Orientation of torso is not going to change!)
        mju_sub3(T_curr_lt_pos, ik_d->sensordata + 7, ik_d->sensordata);           // T(Xl - Xt) (Xt is not going to change!)
        mju_sub3(T_curr_rt_pos, ik_d->sensordata + 14, ik_d->sensordata);          // T(Xr - Xt) (Xt is not going to change!)

        // Update residual: L pos err | L ori err | R pos err | R ori err.
        mju_sub3(residual, T_target_lt_pos, T_curr_lt_pos);      // compute residual
        mju_subQuat(residual + 3, target_lt_quat, curr_lt_quat); // in tgt space
        mju_sub3(residual + 6, T_target_rt_pos, T_curr_rt_pos);  // compute residual
        mju_subQuat(residual + 9, target_rt_quat, curr_rt_quat); // in tgt space

        // Update big J
        J.block<3, 12>(0, 0) = L_JacV.block<3, 12>(0, 6);
        J.block<3, 12>(3, 0) = L_JacW.block<3, 12>(0, 6);
        J.block<3, 12>(6, 0) = R_JacV.block<3, 12>(0, 6);
        J.block<3, 12>(9, 0) = R_JacW.block<3, 12>(0, 6);

        // Solve
        // dqj_nm = J.fullPivHouseholderQr().solve(eig_residual);
        dqj_nm = J.colPivHouseholderQr().solve(eig_residual);
        // dqj_nm = J.ldlt().solve(eig_residual);
        // dqj_nm.block<6,1>(0, 0) = J.block<6, 6>(0, 0).fullPivHouseholderQr().solve(eig_residual.block<6, 1>(0, 0));
        // dqj_nm.block<6,1>(6, 0) = J.block<6, 6>(6, 6).fullPivHouseholderQr().solve(eig_residual.block<6, 1>(6, 0));

        // Compute error norms for early termination
        errorf = eig_residual.norm();
        errorX = dqj_nm.norm();

        // Update qj
        eig_qj_op += dqj_nm;

        itr++;
    }

    // Convergence notification
    if (errorf > ef && errorX > ex)
    {
        solconv = false;
        printf("Sol not converged. Errorf: %lg \t ErrorQ: %lg \t Nitr: %d\n", errorf, errorX, itr);
    }
    else
    {
        mju_copy(qj_op, internal_qj_op, 12);
    }

    // Get the angles back in domain -pi to pi
    // May be problematic?

    // for (int i = 0; i < 12; i++)
    //     qj_op[i] = wrap_angles(qj_op[i]);

    mj_deleteData(ik_d);
}

void bioloid_12dof_IK_position_v3(const mjModel *mm, int tsno, int lfsno, int rfsno,
                                  double global_torso[], double global_torso_kphi[],
                                  double global_leftfoot[], double global_leftfoot_kphi[],
                                  double global_rightfoot[], double global_rightfoot_kphi[],
                                  double qj_op[],
                                  int nitrmax, double ef, double ex)
{
    // Implementation based on T Sugihara's 2011 TRO short paper.
    bool solconv = true;
    /*---------Processing Inputs---Convert axis angle to quat------------------------*/
    double global_torso_kaxis[3] = {global_torso_kphi[0], global_torso_kphi[1], global_torso_kphi[2]};
    double global_leftfoot_kaxis[3] = {global_leftfoot_kphi[0], global_leftfoot_kphi[1], global_leftfoot_kphi[2]};
    double global_rightfoot_kaxis[3] = {global_rightfoot_kphi[0], global_rightfoot_kphi[1], global_rightfoot_kphi[2]};
    mju_normalize3(global_torso_kaxis);
    mju_normalize3(global_leftfoot_kaxis);
    mju_normalize3(global_rightfoot_kaxis);
    double global_torso_quat[4], global_leftfoot_quat[4], global_rightfoot_quat[4];
    mju_axisAngle2Quat(global_torso_quat, global_torso_kaxis, global_torso_kphi[3]);
    mju_axisAngle2Quat(global_leftfoot_quat, global_leftfoot_kaxis, global_leftfoot_kphi[3]);
    mju_axisAngle2Quat(global_rightfoot_quat, global_rightfoot_kaxis, global_rightfoot_kphi[3]);

    /*--make mjData instance for IK problem Jacobian computation----------------------*/
    mjData *ik_d = mj_makeData(mm);
    double z3[3] = {0., 0., 0.}, q_eye4[4] = {1, 0, 0, 0}, internal_qj_op[12];
    // Now the floating base is at origin with default orientation--->
    mju_copy3(ik_d->qpos, z3);
    mju_copy4(ik_d->qpos + 3, q_eye4);
    /*
    All calculations in Torso site frame. It has a constant offset w.r.t. the floating base
    left foot pos and orientation,
    right foot and orientation.
    */

    // Create copy of input estimate and update it. Copy back the answer at last if the iterations converge
    mju_copy(internal_qj_op, qj_op, 12);
    Eigen::Map<Eigen::VectorXd> eig_qj_op(internal_qj_op, 12); // Map Eigen vector to internal_qj_op array

    double target_lt_pos[3], target_rt_pos[3], T_target_lt_pos[3], T_target_rt_pos[3];
    double target_lt_quat[4], target_rt_quat[4];
    double R0T[9], RT0[9];
    mju_quat2Mat(R0T, global_torso_quat);
    mju_transpose(RT0, R0T, 3, 3);
    my_mju_relQuat(target_lt_quat, global_leftfoot_quat, global_torso_quat);  // Set desired orientation
    my_mju_relQuat(target_rt_quat, global_rightfoot_quat, global_torso_quat); // Set desied orientation
    mju_sub3(target_lt_pos, global_leftfoot, global_torso);
    mju_mulMatVec(T_target_lt_pos, RT0, target_lt_pos, 3, 3); // Set desired position in Torso frame: RT0(Xl-Xt)
    mju_sub3(target_rt_pos, global_rightfoot, global_torso);
    mju_mulMatVec(T_target_rt_pos, RT0, target_rt_pos, 3, 3); // Set desired position in Torso frame: RT0(Xr-Xt)

    /*--Compute the linear velocity and angular velocity Jacobian matrices-------------*/
    double l_jacV[3 * 18], l_jacW[3 * 18], r_jacV[3 * 18], r_jacW[3 * 18];

    Eigen::Map<Eigen::Matrix<double, 3, 18, Eigen::RowMajor == 1>> L_JacV(l_jacV, 3, 18);
    Eigen::Map<Eigen::Matrix<double, 3, 18, Eigen::RowMajor == 1>> L_JacW(l_jacW, 3, 18);
    Eigen::Map<Eigen::Matrix<double, 3, 18, Eigen::RowMajor == 1>> R_JacV(r_jacV, 3, 18);
    Eigen::Map<Eigen::Matrix<double, 3, 18, Eigen::RowMajor == 1>> R_JacW(r_jacW, 3, 18);
    Eigen::Matrix<double, 12, 12, Eigen::RowMajor == 1> J;                                                // Matrix gradient of residual(qj)
    Eigen::Matrix<double, 12, 12, Eigen::RowMajor == 1> H;                                                // Jtr We J + Wn
    Eigen::Matrix<double, 12, 12, Eigen::RowMajor == 1> We = Eigen::Matrix<double, 12, 12>::Identity();   // Weight matrix for error
    Eigen::Matrix<double, 12, 12, Eigen::RowMajor == 1> Wn = Eigen::Matrix<double, 12, 12>::Identity();   // Diagonal enrichment matrix
    Eigen::Matrix<double, 12, 12, Eigen::RowMajor == 1> Wnb = Eigen::Matrix<double, 12, 12>::Identity();  // Diagonal matrix of bias values
    Wnb.diagonal() << 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001; // Bias values for each DOF

    double residual[12]; // Equation: residual(qj) = 0 is being solved
    Eigen::Map<Eigen::VectorXd> eig_residual(residual, 12);

    double T_curr_lt_pos[3], T_curr_rt_pos[3], curr_lt_quat[4], curr_rt_quat[4];
    Eigen::VectorXd dqj_nm(12); // Output of Newton's method

    int itr = 0;
    double errorf = 1000, errorX = 1000;
    while (errorf > ef && errorX > ex && itr < nitrmax)
    {
        // Set the new qj for revised computations
        mju_copy(ik_d->qpos + 7, internal_qj_op, 12);
        // Compute mj_step1()-----use mj_kinematics(mm,ik_d)
        mj_step1(mm, ik_d);
        // Compute Jacobian matrices in Torso frame
        mj_jacSite(mm, ik_d, l_jacV, l_jacW, lfsno);
        mj_jacSite(mm, ik_d, r_jacV, r_jacW, rfsno);

        // Read current position and orientation values (These values are in torso frame only becuase ik_d is set to torso frame)
        my_mju_relQuat(curr_lt_quat, ik_d->sensordata + 10, ik_d->sensordata + 3); // Orientation of left foot w.r.t. torso (Orientation of torso is not going to change!)
        my_mju_relQuat(curr_rt_quat, ik_d->sensordata + 17, ik_d->sensordata + 3); // Orientation of right foot w.r.t. torso (Orientation of torso is not going to change!)
        mju_sub3(T_curr_lt_pos, ik_d->sensordata + 7, ik_d->sensordata);           // T(Xl - Xt) (Xt is not going to change!)
        mju_sub3(T_curr_rt_pos, ik_d->sensordata + 14, ik_d->sensordata);          // T(Xr - Xt) (Xt is not going to change!)

        // Update residual: L pos err | L ori err | R pos err | R ori err.
        mju_sub3(residual, T_target_lt_pos, T_curr_lt_pos);      // compute residual
        mju_subQuat(residual + 3, target_lt_quat, curr_lt_quat); // in tgt space
        mju_sub3(residual + 6, T_target_rt_pos, T_curr_rt_pos);  // compute residual
        mju_subQuat(residual + 9, target_rt_quat, curr_rt_quat); // in tgt space

        // Update big J
        J.block<3, 12>(0, 0) = L_JacV.block<3, 12>(0, 6);
        J.block<3, 12>(3, 0) = L_JacW.block<3, 12>(0, 6);
        J.block<3, 12>(6, 0) = R_JacV.block<3, 12>(0, 6);
        J.block<3, 12>(9, 0) = R_JacW.block<3, 12>(0, 6);

        // Solve with T Sugihara's modification to the LM method
        H = J.transpose() * We * J + Wn;
        Wn = eig_residual.transpose() * We * eig_residual * Eigen::Matrix<double, 12, 12>::Identity() + Wnb;
        dqj_nm = H.colPivHouseholderQr().solve(J.transpose() * We * eig_residual);

        // Compute error norms for early termination
        errorf = eig_residual.norm();
        errorX = dqj_nm.norm();

        // Update qj
        eig_qj_op += dqj_nm;

        itr++;
    }

    // Convergence notification
    if (errorf > ef && errorX > ex)
    {
        solconv = false;
        printf("Sol not converged. Errorf: %lg \t ErrorQ: %lg \t Nitr: %d\n", errorf, errorX, itr);
    }
    else
    {
        mju_copy(qj_op, internal_qj_op, 12);
    }

    // Get the angles back in domain -pi to pi
    // May be problematic?

    // for (int i = 0; i < 12; i++)
    //     qj_op[i] = wrap_angles(qj_op[i]);

    mj_deleteData(ik_d);
}

void bioloid_12dof_IK(const mjModel *mm, int tsno, int lfsno, int rfsno,
                      double global_torso[], double global_torso_kphi[],
                      double global_leftfoot[], double global_leftfoot_kphi[],
                      double global_rightfoot[], double global_rightfoot_kphi[],
                      double qj_op[],
                      double global_torso_vel[], double global_torso_omega[],
                      double global_leftfoot_vel[], double global_leftfoot_omega[],
                      double global_rightfoot_vel[], double global_rightfoot_omega[],
                      double dqj_op[],
                      int nitrmax, double ef, double ex)
{
    bool solconv = true;
    /*---------Processing Inputs---Convert axis angle to quat------------------------*/
    double global_torso_kaxis[3] = {global_torso_kphi[0], global_torso_kphi[1], global_torso_kphi[2]};
    double global_leftfoot_kaxis[3] = {global_leftfoot_kphi[0], global_leftfoot_kphi[1], global_leftfoot_kphi[2]};
    double global_rightfoot_kaxis[3] = {global_rightfoot_kphi[0], global_rightfoot_kphi[1], global_rightfoot_kphi[2]};
    mju_normalize3(global_torso_kaxis);
    mju_normalize3(global_leftfoot_kaxis);
    mju_normalize3(global_rightfoot_kaxis);
    double global_torso_quat[4], global_leftfoot_quat[4], global_rightfoot_quat[4];
    mju_axisAngle2Quat(global_torso_quat, global_torso_kaxis, global_torso_kphi[3]);
    mju_axisAngle2Quat(global_leftfoot_quat, global_leftfoot_kaxis, global_leftfoot_kphi[3]);
    mju_axisAngle2Quat(global_rightfoot_quat, global_rightfoot_kaxis, global_rightfoot_kphi[3]);

    /*--make mjData instance for IK problem Jacobian computation----------------------*/
    mjData *ik_d = mj_makeData(mm);
    double z3[3] = {0., 0., 0.}, q_eye4[4] = {1, 0, 0, 0}, internal_qj_op[12];
    // Now the floating base is at origin with default orientation--->
    mju_copy3(ik_d->qpos, z3);
    mju_copy4(ik_d->qpos + 3, q_eye4);
    /*
    All calculations in Torso site frame. It has a constant offset w.r.t. the floating base
    left foot pos and orientation,
    right foot and orientation.
    */

    // Create copy of input estimate and update it. Copy back the answer at last if the iterations converge
    mju_copy(internal_qj_op, qj_op, 12);
    Eigen::Map<Eigen::VectorXd> eig_qj_op(internal_qj_op, 12);

    double target_lt_pos[3], target_rt_pos[3], T_target_lt_pos[3], T_target_rt_pos[3];
    double target_lt_quat[4], target_rt_quat[4];
    double R0T[9], RT0[9];
    mju_quat2Mat(R0T, global_torso_quat);
    mju_transpose(RT0, R0T, 3, 3);
    my_mju_relQuat(target_lt_quat, global_leftfoot_quat, global_torso_quat);  // Set desired orientation
    my_mju_relQuat(target_rt_quat, global_rightfoot_quat, global_torso_quat); // Set desied orientation
    mju_sub3(target_lt_pos, global_leftfoot, global_torso);
    mju_mulMatVec(T_target_lt_pos, RT0, target_lt_pos, 3, 3); // Set desired position in Torso frame: RT0(Xl-Xt)
    mju_sub3(target_rt_pos, global_rightfoot, global_torso);
    mju_mulMatVec(T_target_rt_pos, RT0, target_rt_pos, 3, 3); // Set desired position in Torso frame: RT0(Xr-Xt)

    /*--Compute the linear velocity and angular velocity Jacobian matrices-------------*/
    double l_jacV[3 * 18], l_jacW[3 * 18], r_jacV[3 * 18], r_jacW[3 * 18];

    Eigen::Map<Eigen::Matrix<double, 3, 18, Eigen::RowMajor == 1>> L_JacV(l_jacV, 3, 18);
    Eigen::Map<Eigen::Matrix<double, 3, 18, Eigen::RowMajor == 1>> L_JacW(l_jacW, 3, 18);
    Eigen::Map<Eigen::Matrix<double, 3, 18, Eigen::RowMajor == 1>> R_JacV(r_jacV, 3, 18);
    Eigen::Map<Eigen::Matrix<double, 3, 18, Eigen::RowMajor == 1>> R_JacW(r_jacW, 3, 18);
    Eigen::Matrix<double, 12, 12, Eigen::RowMajor == 1> J; // Matrix gradient of residual(qj)

    double residual[12]; // Equation: residual(qj) = 0 is being solved
    Eigen::Map<Eigen::VectorXd> eig_residual(residual, 12);

    double T_curr_lt_pos[3], T_curr_rt_pos[3], curr_lt_quat[4], curr_rt_quat[4];
    Eigen::VectorXd dqj_nm(12); // Output of Newton's method

    int itr = 0;
    double errorf = 1000, errorX = 1000;
    while (errorf > ef && errorX > ex && itr < nitrmax)
    {
        // Set the new qj for revised computations
        mju_copy(ik_d->qpos + 7, internal_qj_op, 12);
        // Compute mj_step1()
        mj_step1(mm, ik_d);
        // Compute Jacobian matrices in Torso frame
        mj_jacSite(mm, ik_d, l_jacV, l_jacW, lfsno);
        mj_jacSite(mm, ik_d, r_jacV, r_jacW, rfsno);

        // Read current position and orientation values (These values are in torso frame only becuase ik_d is set to torso frame)
        my_mju_relQuat(curr_lt_quat, ik_d->sensordata + 10, ik_d->sensordata + 3); // Orientation of left foot w.r.t. torso (Orientation of torso is not going to change!)
        my_mju_relQuat(curr_rt_quat, ik_d->sensordata + 17, ik_d->sensordata + 3); // Orientation of right foot w.r.t. torso (Orientation of torso is not going to change!)
        mju_sub3(T_curr_lt_pos, ik_d->sensordata + 7, ik_d->sensordata);           // T(Xl - Xt) (Xt is not going to change!)
        mju_sub3(T_curr_rt_pos, ik_d->sensordata + 14, ik_d->sensordata);          // T(Xr - Xt) (Xt is not going to change!)

        // Update residual
        mju_sub3(residual, T_target_lt_pos, T_curr_lt_pos);      // compute residual
        mju_subQuat(residual + 3, target_lt_quat, curr_lt_quat); // in tgt space
        mju_sub3(residual + 6, T_target_rt_pos, T_curr_rt_pos);  // compute residual
        mju_subQuat(residual + 9, target_rt_quat, curr_rt_quat); // in tgt space

        // Update big J
        J.block<3, 12>(0, 0) = L_JacV.block<3, 12>(0, 6);
        J.block<3, 12>(3, 0) = L_JacW.block<3, 12>(0, 6);
        J.block<3, 12>(6, 0) = R_JacV.block<3, 12>(0, 6);
        J.block<3, 12>(9, 0) = R_JacW.block<3, 12>(0, 6);

        // Solve
        dqj_nm = J.fullPivHouseholderQr().solve(eig_residual);
        // dqj_nm.block<6,1>(0, 0) = J.block<6, 6>(0, 0).fullPivHouseholderQr().solve(eig_residual.block<6, 1>(0, 0));
        // dqj_nm.block<6,1>(6, 0) = J.block<6, 6>(6, 6).fullPivHouseholderQr().solve(eig_residual.block<6, 1>(6, 0));

        // Compute error norms for early termination
        errorf = eig_residual.norm();
        errorX = dqj_nm.norm();

        // Update qj
        eig_qj_op += dqj_nm;

        itr++;
    }

    // Convergence notification
    if (errorf > ef && errorX > ex)
    {
        solconv = false;
        printf("Sol not converged. Errorf: %lg \t ErrorQ: %lg \t Nitr: %d\n", errorf, errorX, itr);
    }
    else
    {
        mju_copy(qj_op, internal_qj_op, 12);
    }

    // Get the angles back in domain -pi to pi
    // May be problematic?
    for (int i = 0; i < 12; i++)
        qj_op[i] = wrap_angles(qj_op[i]);

    /* Velocity IK */
    // Set the LHS: {v_lt,w_lt,v_rt,w_rt}, i.e., velocities relative to the Torso frame (floating base)
    double cartesian_vw[12];
    mju_sub3(cartesian_vw, global_leftfoot_vel, global_torso_vel);
    mju_sub3(cartesian_vw + 3, global_leftfoot_omega, global_torso_omega);
    mju_sub3(cartesian_vw + 6, global_rightfoot_vel, global_torso_vel);
    mju_sub3(cartesian_vw + 9, global_rightfoot_omega, global_torso_omega);
    Eigen::Map<Eigen::VectorXd> eig_cartesian_vw(cartesian_vw, 12);
    double internal_dqj_op[12];
    Eigen::Map<Eigen::VectorXd> eig_dqj_op(internal_dqj_op, 12);

    /* Update the ik_d and calculate the Jacobian */
    // The floating base position and orientation have been set already (at the time of IK position)
    // Set the converged qj
    mju_copy(ik_d->qpos + 7, qj_op, 12);
    // Compute mj_step1()
    mj_step1(mm, ik_d);
    // Compute Jacobian matrices
    mj_jacSite(mm, ik_d, l_jacV, l_jacW, lfsno);
    mj_jacSite(mm, ik_d, r_jacV, r_jacW, rfsno);
    // Update big J
    J.block<3, 12>(0, 0) = L_JacV.block<3, 12>(0, 6);
    J.block<3, 12>(3, 0) = L_JacW.block<3, 12>(0, 6);
    J.block<3, 12>(6, 0) = R_JacV.block<3, 12>(0, 6);
    J.block<3, 12>(9, 0) = R_JacW.block<3, 12>(0, 6);
    // Solve for dqj_op
    // Solve
    eig_dqj_op = J.fullPivHouseholderQr().solve(eig_cartesian_vw);
    // Write the output joint rates
    mju_copy(dqj_op, internal_dqj_op, 12);

    mj_deleteData(ik_d);
}

void bioloid_12dof_CoM_IK_position(const mjModel *mm, int tsno, int lfsno, int rfsno,
                                   double global_com[],
                                   double global_torso[], double global_torso_kphi[],
                                   double global_leftfoot[], double global_leftfoot_kphi[],
                                   double global_rightfoot[], double global_rightfoot_kphi[],
                                   double qj_op[],
                                   int nitrmax, double ef, double ex)
{
    bool solconv = true;
    /*---------Processing Inputs---Convert axis angle to quat------------------------*/
    double global_torso_kaxis[3] = {global_torso_kphi[0], global_torso_kphi[1], global_torso_kphi[2]};
    double global_leftfoot_kaxis[3] = {global_leftfoot_kphi[0], global_leftfoot_kphi[1], global_leftfoot_kphi[2]};
    double global_rightfoot_kaxis[3] = {global_rightfoot_kphi[0], global_rightfoot_kphi[1], global_rightfoot_kphi[2]};
    mju_normalize3(global_torso_kaxis);
    mju_normalize3(global_leftfoot_kaxis);
    mju_normalize3(global_rightfoot_kaxis);
    double global_torso_quat[4], global_leftfoot_quat[4], global_rightfoot_quat[4];
    mju_axisAngle2Quat(global_torso_quat, global_torso_kaxis, global_torso_kphi[3]);
    mju_axisAngle2Quat(global_leftfoot_quat, global_leftfoot_kaxis, global_leftfoot_kphi[3]);
    mju_axisAngle2Quat(global_rightfoot_quat, global_rightfoot_kaxis, global_rightfoot_kphi[3]);

    /*--make mjData instance for IK problem Jacobian computation----------------------*/
    mjData *ik_d = mj_makeData(mm);
    double z3[3] = {0., 0., 0.}, q_eye4[4] = {1, 0, 0, 0}, internal_qj_op[12];
    // Now the floating base is at origin with default orientation--->
    mju_copy3(ik_d->qpos, z3);
    mju_copy4(ik_d->qpos + 3, q_eye4);
    /*
    All calculations in Torso site frame. It has a constant offset w.r.t. the floating base
    left foot pos and orientation,
    right foot and orientation.
    */

    // Create copy of input estimate and update it. Copy back the answer at last if iterations converge
    mju_copy(internal_qj_op, qj_op, 12);
    Eigen::Map<Eigen::VectorXd> eig_qj_op(internal_qj_op, 12);

    // Control COM-L & COM-R positions and T-L & T-R orientation
    double target_lg_pos[3], target_rg_pos[3], T_target_lg_pos[3], T_target_rg_pos[3];
    double target_lt_quat[4], target_rt_quat[4];
    double R0T[9], RT0[9];
    mju_quat2Mat(R0T, global_torso_quat);
    mju_transpose(RT0, R0T, 3, 3);
    my_mju_relQuat(target_lt_quat, global_leftfoot_quat, global_torso_quat);  // Set desired orientation
    my_mju_relQuat(target_rt_quat, global_rightfoot_quat, global_torso_quat); // Set desied orientation
    mju_sub3(target_lg_pos, global_leftfoot, global_com);
    mju_mulMatVec(T_target_lg_pos, RT0, target_lg_pos, 3, 3); // Set desired position in Torso frame: RT0(Xl-Xg)
    mju_sub3(target_rg_pos, global_rightfoot, global_com);
    mju_mulMatVec(T_target_rg_pos, RT0, target_rg_pos, 3, 3); // Set desired position in Torso frame: RT0(Xr-Xg)

    /*--Compute the linear velocity and angular velocity Jacobian matrices-------------*/
    double l_jacV[3 * 18], l_jacW[3 * 18], r_jacV[3 * 18], r_jacW[3 * 18];
    double g_jacV[3 * 18];
    double lg_jacV[3 * 18], rg_jacV[3 * 18];
    Eigen::Map<Eigen::Matrix<double, 3, 18, Eigen::RowMajor == 1>> LG_JacV(lg_jacV, 3, 18);
    Eigen::Map<Eigen::Matrix<double, 3, 18, Eigen::RowMajor == 1>> L_JacW(l_jacW, 3, 18);
    Eigen::Map<Eigen::Matrix<double, 3, 18, Eigen::RowMajor == 1>> RG_JacV(rg_jacV, 3, 18);
    Eigen::Map<Eigen::Matrix<double, 3, 18, Eigen::RowMajor == 1>> R_JacW(r_jacW, 3, 18);
    Eigen::Matrix<double, 12, 12, Eigen::RowMajor == 1> J; // Matrix gradient of residual(qj)

    double residual[12]; // Equation: residual(qj) = 0 is being solved
    Eigen::Map<Eigen::VectorXd> eig_residual(residual, 12);

    double T_curr_lg_pos[3], T_curr_rg_pos[3], curr_lt_quat[4], curr_rt_quat[4];
    Eigen::VectorXd dqj_nm(12); // Output of Newton's method

    int itr = 0;
    double errorf = 1000, errorX = 1000;
    while (errorf > ef && errorX > ex && itr < 20)
    {
        // Set the new qj for revised computations
        mju_copy(ik_d->qpos + 7, internal_qj_op, 12);
        // Compute mj_step1()
        mj_step1(mm, ik_d);
        // Compute Jacobian matrices in Torso frame
        mj_jacSite(mm, ik_d, l_jacV, l_jacW, lfsno);
        mj_jacSite(mm, ik_d, r_jacV, r_jacW, rfsno);
        mj_jacSubtreeCom(mm, ik_d, g_jacV, 1);
        mju_sub(lg_jacV, l_jacV, g_jacV, 54);
        mju_sub(rg_jacV, r_jacV, g_jacV, 54);

        // Read current position and orientation values (These values are in torso frame only becuase ik_d is set to torso frame)
        my_mju_relQuat(curr_lt_quat, ik_d->sensordata + 10, ik_d->sensordata + 3); // Orientation of left foot w.r.t. torso (Orientation of torso is not going to change!)
        my_mju_relQuat(curr_rt_quat, ik_d->sensordata + 17, ik_d->sensordata + 3); // Orientation of right foot w.r.t. torso (Orientation of torso is not going to change!)
        mju_sub3(T_curr_lg_pos, ik_d->sensordata + 7, ik_d->sensordata + 39);      // T(Xl - Xg)
        mju_sub3(T_curr_rg_pos, ik_d->sensordata + 14, ik_d->sensordata + 39);     // T(Xr - Xg)

        // Update residual
        mju_sub3(residual, T_target_lg_pos, T_curr_lg_pos);      // compute residual
        mju_subQuat(residual + 3, target_lt_quat, curr_lt_quat); // in tgt space
        mju_sub3(residual + 6, T_target_rg_pos, T_curr_rg_pos);  // compute residual
        mju_subQuat(residual + 9, target_rt_quat, curr_rt_quat); // in tgt space

        // Update big J
        J.block<3, 12>(0, 0) = LG_JacV.block<3, 12>(0, 6);
        J.block<3, 12>(3, 0) = L_JacW.block<3, 12>(0, 6);
        J.block<3, 12>(6, 0) = RG_JacV.block<3, 12>(0, 6);
        J.block<3, 12>(9, 0) = R_JacW.block<3, 12>(0, 6);

        // Solve
        dqj_nm = J.fullPivHouseholderQr().solve(eig_residual);

        // Compute error norms for early termination
        errorf = eig_residual.norm();
        errorX = dqj_nm.norm();
        // Update qj
        eig_qj_op += dqj_nm;

        itr++;
    }

    // Convergence notification
    if (errorf > ef && errorX > ex)
    {
        solconv = false;
        printf("Sol not converged. Errorf: %lg \t ErrorQ: %lg \t Nitr: %d\n", errorf, errorX, itr);
    }
    else
    {
        mju_copy(qj_op, internal_qj_op, 12);
    }

    // Get the angles back in domain -pi to pi
    // May be problematic?
    // for (int i = 0; i < 12; i++)
    //     qj_op[i] = wrap_angles(qj_op[i]);

    mj_deleteData(ik_d);
}

void bioloid_12dof_CoM_IK(const mjModel *mm, int tsno, int lfsno, int rfsno,
                          double global_com[],
                          double global_torso[], double global_torso_kphi[],
                          double global_leftfoot[], double global_leftfoot_kphi[],
                          double global_rightfoot[], double global_rightfoot_kphi[],
                          double qj_op[],
                          double global_com_vel[],
                          double global_torso_omega[],
                          double global_leftfoot_vel[], double global_leftfoot_omega[],
                          double global_rightfoot_vel[], double global_rightfoot_omega[],
                          double dqj_op[],
                          int nitrmax, double ef, double ex)
{
    bool solconv = true;
    /*---------Processing Inputs---Convert axis angle to quat------------------------*/
    double global_torso_kaxis[3] = {global_torso_kphi[0], global_torso_kphi[1], global_torso_kphi[2]};
    double global_leftfoot_kaxis[3] = {global_leftfoot_kphi[0], global_leftfoot_kphi[1], global_leftfoot_kphi[2]};
    double global_rightfoot_kaxis[3] = {global_rightfoot_kphi[0], global_rightfoot_kphi[1], global_rightfoot_kphi[2]};
    mju_normalize3(global_torso_kaxis);
    mju_normalize3(global_leftfoot_kaxis);
    mju_normalize3(global_rightfoot_kaxis);
    double global_torso_quat[4], global_leftfoot_quat[4], global_rightfoot_quat[4];
    mju_axisAngle2Quat(global_torso_quat, global_torso_kaxis, global_torso_kphi[3]);
    mju_axisAngle2Quat(global_leftfoot_quat, global_leftfoot_kaxis, global_leftfoot_kphi[3]);
    mju_axisAngle2Quat(global_rightfoot_quat, global_rightfoot_kaxis, global_rightfoot_kphi[3]);

    /*--make mjData instance for IK problem Jacobian computation----------------------*/
    mjData *ik_d = mj_makeData(mm);
    double z3[3] = {0., 0., 0.}, q_eye4[4] = {1, 0, 0, 0}, internal_qj_op[12];
    // Now the floating base is at origin with default orientation--->
    mju_copy3(ik_d->qpos, z3);
    mju_copy4(ik_d->qpos + 3, q_eye4);
    /*
    All calculations in Torso site frame. It has a constant offset w.r.t. the floating base
    left foot pos and orientation,
    right foot and orientation.
    */

    // Create copy of input estimate and update it. Copy back the answer at last if iterations converge
    mju_copy(internal_qj_op, qj_op, 12);
    Eigen::Map<Eigen::VectorXd> eig_qj_op(internal_qj_op, 12);

    // Control COM-L & COM-R positions and T-L & T-R orientation
    double target_lg_pos[3], target_rg_pos[3], T_target_lg_pos[3], T_target_rg_pos[3];
    double target_lt_quat[4], target_rt_quat[4];
    double R0T[9], RT0[9];
    mju_quat2Mat(R0T, global_torso_quat);
    mju_transpose(RT0, R0T, 3, 3);
    my_mju_relQuat(target_lt_quat, global_leftfoot_quat, global_torso_quat);  // Set desired orientation
    my_mju_relQuat(target_rt_quat, global_rightfoot_quat, global_torso_quat); // Set desied orientation
    mju_sub3(target_lg_pos, global_leftfoot, global_com);
    mju_mulMatVec(T_target_lg_pos, RT0, target_lg_pos, 3, 3); // Set desired position in Torso frame: RT0(Xl-Xg)
    mju_sub3(target_rg_pos, global_rightfoot, global_com);
    mju_mulMatVec(T_target_rg_pos, RT0, target_rg_pos, 3, 3); // Set desired position in Torso frame: RT0(Xr-Xg)

    /*--Compute the linear velocity and angular velocity Jacobian matrices-------------*/
    double l_jacV[3 * 18], l_jacW[3 * 18], r_jacV[3 * 18], r_jacW[3 * 18];
    double g_jacV[3 * 18];
    double lg_jacV[3 * 18], rg_jacV[3 * 18];
    Eigen::Map<Eigen::Matrix<double, 3, 18, Eigen::RowMajor == 1>> LG_JacV(lg_jacV, 3, 18);
    Eigen::Map<Eigen::Matrix<double, 3, 18, Eigen::RowMajor == 1>> L_JacW(l_jacW, 3, 18);
    Eigen::Map<Eigen::Matrix<double, 3, 18, Eigen::RowMajor == 1>> RG_JacV(rg_jacV, 3, 18);
    Eigen::Map<Eigen::Matrix<double, 3, 18, Eigen::RowMajor == 1>> R_JacW(r_jacW, 3, 18);
    Eigen::Matrix<double, 12, 12, Eigen::RowMajor == 1> J; // Matrix gradient of residual(qj)

    double residual[12]; // Equation: residual(qj) = 0 is being solved
    Eigen::Map<Eigen::VectorXd> eig_residual(residual, 12);

    double T_curr_lg_pos[3], T_curr_rg_pos[3], curr_lt_quat[4], curr_rt_quat[4];
    Eigen::VectorXd dqj_nm(12); // Output of Newton's method

    int itr = 0;
    double errorf = 1000, errorX = 1000;
    while (errorf > ef && errorX > ex && itr < 20)
    {
        // Set the new qj for revised computations
        mju_copy(ik_d->qpos + 7, internal_qj_op, 12);
        // Compute mj_step1()
        mj_step1(mm, ik_d);
        // Compute Jacobian matrices in Torso frame
        mj_jacSite(mm, ik_d, l_jacV, l_jacW, lfsno);
        mj_jacSite(mm, ik_d, r_jacV, r_jacW, rfsno);
        mj_jacSubtreeCom(mm, ik_d, g_jacV, 1);
        mju_sub(lg_jacV, l_jacV, g_jacV, 54);
        mju_sub(rg_jacV, r_jacV, g_jacV, 54);

        // Read current position and orientation values (These values are in torso frame only becuase ik_d is set to torso frame)
        my_mju_relQuat(curr_lt_quat, ik_d->sensordata + 10, ik_d->sensordata + 3); // Orientation of left foot w.r.t. torso (Orientation of torso is not going to change!)
        my_mju_relQuat(curr_rt_quat, ik_d->sensordata + 17, ik_d->sensordata + 3); // Orientation of right foot w.r.t. torso (Orientation of torso is not going to change!)
        mju_sub3(T_curr_lg_pos, ik_d->sensordata + 7, ik_d->sensordata + 39);      // T(Xl - Xg)
        mju_sub3(T_curr_rg_pos, ik_d->sensordata + 14, ik_d->sensordata + 39);     // T(Xr - Xg)

        // Update residual
        mju_sub3(residual, T_target_lg_pos, T_curr_lg_pos);      // compute residual
        mju_subQuat(residual + 3, target_lt_quat, curr_lt_quat); // in tgt space
        mju_sub3(residual + 6, T_target_rg_pos, T_curr_rg_pos);  // compute residual
        mju_subQuat(residual + 9, target_rt_quat, curr_rt_quat); // in tgt space

        // Update big J
        J.block<3, 12>(0, 0) = LG_JacV.block<3, 12>(0, 6);
        J.block<3, 12>(3, 0) = L_JacW.block<3, 12>(0, 6);
        J.block<3, 12>(6, 0) = RG_JacV.block<3, 12>(0, 6);
        J.block<3, 12>(9, 0) = R_JacW.block<3, 12>(0, 6);

        // Solve
        dqj_nm = J.fullPivHouseholderQr().solve(eig_residual);

        // Compute error norms for early termination
        errorf = eig_residual.norm();
        errorX = dqj_nm.norm();
        // Update qj
        eig_qj_op += dqj_nm;

        itr++;
    }

    // Convergence notification
    if (errorf > ef && errorX > ex)
    {
        solconv = false;
        printf("Sol not converged. Errorf: %lg \t ErrorQ: %lg \t Nitr: %d\n", errorf, errorX, itr);
    }
    else
    {
        mju_copy(qj_op, internal_qj_op, 12);
    }

    /* Velocity COM-IK */
    // Set the LHS: {v_lg,w_lt,v_rg,w_rt}
    double cartesian_vw[12];
    mju_sub3(cartesian_vw, global_leftfoot_vel, global_com_vel);
    mju_sub3(cartesian_vw + 3, global_leftfoot_omega, global_torso_omega);
    mju_sub3(cartesian_vw + 6, global_rightfoot_vel, global_com_vel);
    mju_sub3(cartesian_vw + 9, global_rightfoot_omega, global_torso_omega);
    Eigen::Map<Eigen::VectorXd> eig_cartesian_vw(cartesian_vw, 12);
    double internal_dqj_op[12];
    Eigen::Map<Eigen::VectorXd> eig_dqj_op(internal_dqj_op, 12);

    /* Update the ik_d and calculate the Jacobian */
    // The floating base position and orientation have been set already (at the time of IK position)
    // Set the converged qj
    mju_copy(ik_d->qpos + 7, qj_op, 12);
    // Compute mj_step1()
    mj_step1(mm, ik_d);
    // Compute Jacobian matrices in Torso frame
    mj_jacSite(mm, ik_d, l_jacV, l_jacW, lfsno);
    mj_jacSite(mm, ik_d, r_jacV, r_jacW, rfsno);
    mj_jacSubtreeCom(mm, ik_d, g_jacV, 1);
    mju_sub(lg_jacV, l_jacV, g_jacV, 54);
    mju_sub(rg_jacV, r_jacV, g_jacV, 54);
    // Update big J
    J.block<3, 12>(0, 0) = LG_JacV.block<3, 12>(0, 6);
    J.block<3, 12>(3, 0) = L_JacW.block<3, 12>(0, 6);
    J.block<3, 12>(6, 0) = RG_JacV.block<3, 12>(0, 6);
    J.block<3, 12>(9, 0) = R_JacW.block<3, 12>(0, 6);
    // Solve for dqj_op
    // Solve
    eig_dqj_op = J.fullPivHouseholderQr().solve(eig_cartesian_vw);
    // Write the output joint rates
    mju_copy(dqj_op, internal_dqj_op, 12);

    mj_deleteData(ik_d);
}

void kondo_12dof_IK_position(const mjModel *mm, int tsno, int lfsno, int rfsno,
                             double global_torso[], double global_torso_kphi[],
                             double global_leftfoot[], double global_leftfoot_kphi[],
                             double global_rightfoot[], double global_rightfoot_kphi[],
                             double qj_op[],
                             int nitrmax, double ef, double ex)
{
    // No processing on the last 10 DOF of Kondo qj_op

    bool llsolconv = true, rlsolconv = true;
    /*---------Processing Inputs---Convert axis angle to quat------------------------*/
    double global_torso_quat[4], global_leftfoot_quat[4], global_rightfoot_quat[4];
    double global_torso_kaxis[3] = {global_torso_kphi[0], global_torso_kphi[1], global_torso_kphi[2]};
    double global_leftfoot_kaxis[3] = {global_leftfoot_kphi[0], global_leftfoot_kphi[1], global_leftfoot_kphi[2]};
    double global_rightfoot_kaxis[3] = {global_rightfoot_kphi[0], global_rightfoot_kphi[1], global_rightfoot_kphi[2]};
    mju_normalize3(global_torso_kaxis);
    mju_normalize3(global_leftfoot_kaxis);
    mju_normalize3(global_rightfoot_kaxis);
    mju_axisAngle2Quat(global_torso_quat, global_torso_kaxis, global_torso_kphi[3]);
    mju_axisAngle2Quat(global_leftfoot_quat, global_leftfoot_kaxis, global_leftfoot_kphi[3]);
    mju_axisAngle2Quat(global_rightfoot_quat, global_rightfoot_kaxis, global_rightfoot_kphi[3]);

    /*--make mjData instance for IK problem Jacobian computation----------------------*/
    mjData *ik_d = mj_makeData(mm);

    // Position of the site w.r.t the floating base
    double torso_site_offsetpos[3], torso_site_offsetquat[4];
    mju_copy3(torso_site_offsetpos, mm->site_pos + tsno * 3);   // Correct
    mju_copy4(torso_site_offsetquat, mm->site_quat + tsno * 4); // Correct

    // Solve for the floating base orientation
    my_mju_relQuat(ik_d->qpos + 3, global_torso_quat, torso_site_offsetquat); // Correct
    // Solve for the floating base position given the torso site position and orientation
    double rot_fb[9];
    mju_quat2Mat(rot_fb, ik_d->qpos + 3);
    mju_mulMatVec(ik_d->qpos, rot_fb, torso_site_offsetpos, 3, 3);
    mju_sub3(ik_d->qpos, global_torso, ik_d->qpos);
    //  Rest of the joints qj = [qjL|qjR| uppderbody DOFs]
    mju_copy(ik_d->qpos + 7, qj_op, 22);

    // Turn off collision detection for this mjData??
    // mj_step1(mm, ik_d);
    mj_step1(mm, ik_d);

    // printf("IK function q\n");
    // for (int i = 0; i < mm->nq; i++)
    //     printf("%.4f\t", ik_d->qpos[i]);
    // printf("\n\n");

    /*--Compute the linear velocity and angular velocity Jacobian matrices-------------*/
    // Jacobian dimensions are higher: Jv is 3x(22+6) and Jw is 3x(22+6).
    double ll_jacV[3 * 28], ll_jacW[3 * 28], rl_jacV[3 * 28], rl_jacW[3 * 28];
    mj_jacSite(mm, ik_d, ll_jacV, ll_jacW, lfsno);
    mj_jacSite(mm, ik_d, rl_jacV, rl_jacW, rfsno);
    // Full Jacobians mapped
    Eigen::Map<Eigen::Matrix<double, 3, 28, Eigen::RowMajor == 1>> LL_JacV(ll_jacV, 3, 28);
    Eigen::Map<Eigen::Matrix<double, 3, 28, Eigen::RowMajor == 1>> LL_JacW(ll_jacW, 3, 28);
    Eigen::Map<Eigen::Matrix<double, 3, 28, Eigen::RowMajor == 1>> RL_JacV(rl_jacV, 3, 28);
    Eigen::Map<Eigen::Matrix<double, 3, 28, Eigen::RowMajor == 1>> RL_JacW(rl_jacW, 3, 28);
    // Pick out only left leg and right leg parts
    // Jjl is a 6x6 matrix mapping qjl_dot to [Vlx Vly Vlz Wlx Wly Wlz]
    // Jjr is a 6x6 matrix mapping qjr_dot to [Vrx Vry Vrz Wrx Wry Wrz]
    Eigen::Matrix<double, 6, 6, Eigen::RowMajor == 1> Jjl;
    Jjl.block<3, 6>(0, 0) = LL_JacV.block<3, 6>(0, 6);
    Jjl.block<3, 6>(3, 0) = LL_JacW.block<3, 6>(0, 6);
    Eigen::Matrix<double, 6, 6, Eigen::RowMajor == 1> Jjr;
    Jjr.block<3, 6>(0, 0) = RL_JacV.block<3, 6>(0, 12);
    Jjr.block<3, 6>(3, 0) = RL_JacW.block<3, 6>(0, 12);

    /*---Solving the inverse kinematics problem for the left leg------------------*/
    int itr = 0;
    double errorf = 1000, errorX = 1000;
    double ll_residual[6], qjl_op[6];
    Eigen::Map<Eigen::VectorXd> eig_ll_residual(ll_residual, 6);
    mju_copy(qjl_op, qj_op, 6);
    Eigen::Map<Eigen::VectorXd> eig_qjl_op(qjl_op, 6);
    Eigen::VectorXd dql_nm(6); // dql_nm: dq generated for the left leg using Newton's Method

    double left_foot_site_curr[7], left_foot_site_target[7];    // Left foot site pose: [x y z | e0 e1 e2 e3]
    mju_copy3(left_foot_site_target, global_leftfoot);          // built from function inputs
    mju_copy4(left_foot_site_target + 3, global_leftfoot_quat); // built from function inputs

    // mj_step1 has been computed
    while (errorf > ef && errorX > ex && itr < nitrmax)
    {
        // compute left foot site frame current pos // 0-1-2 3-4-5-6 are torso site
        mju_copy(left_foot_site_curr, ik_d->sensordata + 7, 7); // 7-8-9 10-11-12-13 are left foot
        // compute residual6x1 = Current frame pose [x y z | e0 e1 e2 e3] - Target frame pose
        mju_sub3(ll_residual, left_foot_site_curr, left_foot_site_target);
        mju_subQuat(ll_residual + 3, left_foot_site_curr + 3, left_foot_site_target + 3);

        // mju_printMat(left_foot_site_curr, 1, 7);
        // mju_printMat(left_foot_site_target, 1, 7);
        // std::cout << "residual:" << eig_residual.transpose() << std::endl;

        // errorf is norm(residual)
        // This is problematic because of three linear velocities and three angular velocities
        errorf = eig_ll_residual.norm();

        // dq_nm = -Jjl.inverse() * eig_residual;
        dql_nm = Jjl.fullPivHouseholderQr().solve(-eig_ll_residual);
        eig_qjl_op += dql_nm;

        errorX = dql_nm.norm();

        // Write to ik_d data
        mju_copy(ik_d->qpos + 7, qjl_op, 6);
        // mj_step1() with the updated solution of qjl
        // mj_step1(mm, ik_d);
        mj_step1(mm, ik_d);
        // Compute Jacobian with updated qjl
        mj_jacSite(mm, ik_d, ll_jacV, ll_jacW, 1); // Left foot site ID = 1

        // Update the mapped Eigen matrices: LL_JacV, LL_JacW-----No need. They get updated automatically
        // Update Jjl
        Jjl.block<3, 6>(0, 0) = LL_JacV.block<3, 6>(0, 6);
        Jjl.block<3, 6>(3, 0) = LL_JacW.block<3, 6>(0, 6);

        // printf("Left foot IK: errorf: %lg \t errorQ: %lg \t iterations: %d\n", errorf, errorX, itr);
        //   mju_printMat(ik_d->qpos,19,1);

        itr++;
    }

    // Convergence notification
    if (errorf > ef && errorX > ex)
    {
        llsolconv = false;
        printf("LL sol not converged. Errorf: %lg \t ErrorQ: %lg \t Nitr: %d\n", errorf, errorX, itr);
    }
    else
        mju_copy(qj_op, qjl_op, 6);

    /*---Solving the inverse kinematics problem for the right leg------------------*/
    errorf = 1000;
    errorX = 1000;
    itr = 0;
    double rl_residual[6], qjr_op[6];
    Eigen::Map<Eigen::VectorXd> eig_rl_residual(rl_residual, 6);
    mju_copy(qjr_op, qj_op + 6, 6); // last 6 elements are the right leg joint angles
    Eigen::Map<Eigen::VectorXd> eig_qjr_op(qjr_op, 6);
    Eigen::VectorXd dqr_nm(6); // dqr_nm: dq generated for the right leg using Newton's Method

    double right_foot_site_curr[7], right_foot_site_target[7];    // Right foot site pose: [x y z | e0 e1 e2 e3]
    mju_copy3(right_foot_site_target, global_rightfoot);          // built from function inputs
    mju_copy4(right_foot_site_target + 3, global_rightfoot_quat); // built from function inputs

    // mj_step1 has been computed
    // printf("\n");
    while (errorf > ef && errorX > ex && itr < nitrmax)
    {
        // compute right foot site frame current pos // 0-1-2 3-4-5-6 are torso site
        mju_copy(right_foot_site_curr, ik_d->sensordata + 7 + 7, 7); // 14-15-16 17-18-19-20 are right foot
        // compute residual6x1 = Current frame pose [x y z | e0 e1 e2 e3] - Target frame pose
        mju_sub3(rl_residual, right_foot_site_curr, right_foot_site_target);
        mju_subQuat(rl_residual + 3, right_foot_site_curr + 3, right_foot_site_target + 3);

        /*         mju_printMat(right_foot_site_curr, 1, 7);
                mju_printMat(right_foot_site_target, 1, 7);
                mju_printMat(rl_residual, 1, 6); */
        // std::cout << "residual:" << eig_rl_residual.transpose() << std::endl;

        // errorf is norm(residual)
        // This is problematic because of three linear velocities and three angular velocities
        errorf = eig_rl_residual.norm();

        // dq_nm = -Jjr.inverse() * eig_rl_esidue;
        dqr_nm = Jjr.fullPivHouseholderQr().solve(-eig_rl_residual);
        eig_qjr_op += dqr_nm;

        errorX = dqr_nm.norm();

        // Write to ik_d data
        mju_copy(ik_d->qpos + 7 + 6, qjr_op, 6);
        // mj_step1() with the updated solution of qjr
        mj_step1(mm, ik_d);
        // Compute Jacobian with updated qjr
        mj_jacSite(mm, ik_d, rl_jacV, rl_jacW, 2); // Right foot site ID = 2

        // Update the mapped Eigen matrices: RL_JacV, RL_JacW-----No need. They get updated automatically
        // Update Jjr
        Jjr.block<3, 6>(0, 0) = RL_JacV.block<3, 6>(0, 12);
        Jjr.block<3, 6>(3, 0) = RL_JacW.block<3, 6>(0, 12);

        // printf("Right leg IK: errorf: %lg \t errorQ: %lg \t iterations: %d\n", errorf, errorX, itr);
        //  mju_printMat(ik_d->qpos,19,1);

        itr++;
    }

    // Convergence notification
    if (errorf > ef && errorX > ex)
    {
        rlsolconv = false;
        printf("RL sol not converged. Errorf: %lg \t ErrorQ: %lg \t Nitr: %d\n", errorf, errorX, itr);
    }
    else
        mju_copy(qj_op + 6, qjr_op, 6);

    // Get the angles back in domain -pi to pi
    // May be problematic?
    for (int i = 0; i < 12; i++)
        qj_op[i] = wrap_angles(qj_op[i]);

    mj_deleteData(ik_d);

    // Debugging:
    /*     printf("\n Debugging q\n");
        for (int i = 0; i < 12; i++)
            printf("%.4f,", qj_op[i]);
        printf("\n");
        printf("\n Debugging q\n");
        for (int i = 0; i < 12; i++)
            printf("%.4f\t", qj_op[i]);
        printf("\n\n"); */
}

void kondo_12dof_IK_position_v2(const mjModel *mm, int tsno, int lfsno, int rfsno,
                                double global_torso[], double global_torso_kphi[],
                                double global_leftfoot[], double global_leftfoot_kphi[],
                                double global_rightfoot[], double global_rightfoot_kphi[],
                                double qj_op[],
                                int nitrmax, double ef, double ex)
{
    // No processing on the last 10 DOF of Kondo qj_op

    bool solconv = true;
    /*---------Processing Inputs---Convert axis angle to quat------------------------*/
    double global_torso_kaxis[3] = {global_torso_kphi[0], global_torso_kphi[1], global_torso_kphi[2]};
    double global_leftfoot_kaxis[3] = {global_leftfoot_kphi[0], global_leftfoot_kphi[1], global_leftfoot_kphi[2]};
    double global_rightfoot_kaxis[3] = {global_rightfoot_kphi[0], global_rightfoot_kphi[1], global_rightfoot_kphi[2]};
    mju_normalize3(global_torso_kaxis);
    mju_normalize3(global_leftfoot_kaxis);
    mju_normalize3(global_rightfoot_kaxis);
    double global_torso_quat[4], global_leftfoot_quat[4], global_rightfoot_quat[4];
    mju_axisAngle2Quat(global_torso_quat, global_torso_kaxis, global_torso_kphi[3]);
    mju_axisAngle2Quat(global_leftfoot_quat, global_leftfoot_kaxis, global_leftfoot_kphi[3]);
    mju_axisAngle2Quat(global_rightfoot_quat, global_rightfoot_kaxis, global_rightfoot_kphi[3]);

    /*--make mjData instance for IK problem Jacobian computation----------------------*/
    mjData *ik_d = mj_makeData(mm);
    double z3[3] = {0., 0., 0.}, q_eye4[4] = {1, 0, 0, 0}, internal_qj_op[12];
    // Now the floating base is at origin with default orientation--->
    mju_copy3(ik_d->qpos, z3);
    mju_copy4(ik_d->qpos + 3, q_eye4);
    /*
    All calculations in Torso site frame. It has a constant offset w.r.t. the floating base
    left foot pos and orientation,
    right foot and orientation.
    */

    // Create copy of first 12 DOF of the input estimate and update it. Copy back the answer at last if the iterations converge
    mju_copy(internal_qj_op, qj_op, 12);
    Eigen::Map<Eigen::VectorXd> eig_qj_op(internal_qj_op, 12);

    double target_lt_pos[3], target_rt_pos[3], T_target_lt_pos[3], T_target_rt_pos[3];
    double target_lt_quat[4], target_rt_quat[4];
    double R0T[9], RT0[9];
    mju_quat2Mat(R0T, global_torso_quat);
    mju_transpose(RT0, R0T, 3, 3);
    my_mju_relQuat(target_lt_quat, global_leftfoot_quat, global_torso_quat);  // Set desired orientation
    my_mju_relQuat(target_rt_quat, global_rightfoot_quat, global_torso_quat); // Set desied orientation
    mju_sub3(target_lt_pos, global_leftfoot, global_torso);
    mju_mulMatVec(T_target_lt_pos, RT0, target_lt_pos, 3, 3); // Set desired position in Torso frame: RT0(Xl-Xt)
    mju_sub3(target_rt_pos, global_rightfoot, global_torso);
    mju_mulMatVec(T_target_rt_pos, RT0, target_rt_pos, 3, 3); // Set desired position in Torso frame: RT0(Xr-Xt)

    /*--Compute the linear velocity and angular velocity Jacobian matrices-------------*/
    double l_jacV[3 * 28], l_jacW[3 * 28], r_jacV[3 * 28], r_jacW[3 * 28];

    Eigen::Map<Eigen::Matrix<double, 3, 28, Eigen::RowMajor == 1>> L_JacV(l_jacV, 3, 28);
    Eigen::Map<Eigen::Matrix<double, 3, 28, Eigen::RowMajor == 1>> L_JacW(l_jacW, 3, 28);
    Eigen::Map<Eigen::Matrix<double, 3, 28, Eigen::RowMajor == 1>> R_JacV(r_jacV, 3, 28);
    Eigen::Map<Eigen::Matrix<double, 3, 28, Eigen::RowMajor == 1>> R_JacW(r_jacW, 3, 28);
    Eigen::Matrix<double, 12, 12, Eigen::RowMajor == 1> J; // Matrix gradient of residual(qj)

    double residual[12]; // Equation: residual(qj) = 0 is being solved
    Eigen::Map<Eigen::VectorXd> eig_residual(residual, 12);

    double T_curr_lt_pos[3], T_curr_rt_pos[3], curr_lt_quat[4], curr_rt_quat[4];
    Eigen::VectorXd dqj_nm(12); // Output of Newton's method

    int itr = 0;
    double errorf = 1000, errorX = 1000;
    while (errorf > ef && errorX > ex && itr < nitrmax)
    {
        // Set the new qj for revised computations
        mju_copy(ik_d->qpos + 7, internal_qj_op, 12);
        // Compute mj_step1()
        mj_step1(mm, ik_d);
        // Compute Jacobian matrices in Torso frame
        mj_jacSite(mm, ik_d, l_jacV, l_jacW, lfsno);
        mj_jacSite(mm, ik_d, r_jacV, r_jacW, rfsno);

        // Read current position and orientation values (These values are in torso frame only becuase ik_d is set to torso frame)
        my_mju_relQuat(curr_lt_quat, ik_d->sensordata + 10, ik_d->sensordata + 3); // Orientation of left foot w.r.t. torso (Orientation of torso is not going to change!)
        my_mju_relQuat(curr_rt_quat, ik_d->sensordata + 17, ik_d->sensordata + 3); // Orientation of right foot w.r.t. torso (Orientation of torso is not going to change!)
        mju_sub3(T_curr_lt_pos, ik_d->sensordata + 7, ik_d->sensordata);           // T(Xl - Xt) (Xt is not going to change!)
        mju_sub3(T_curr_rt_pos, ik_d->sensordata + 14, ik_d->sensordata);          // T(Xr - Xt) (Xt is not going to change!)

        // Update residual
        mju_sub3(residual, T_target_lt_pos, T_curr_lt_pos);      // compute residual
        mju_subQuat(residual + 3, target_lt_quat, curr_lt_quat); // in tgt space
        mju_sub3(residual + 6, T_target_rt_pos, T_curr_rt_pos);  // compute residual
        mju_subQuat(residual + 9, target_rt_quat, curr_rt_quat); // in tgt space

        // Update big J
        J.block<3, 12>(0, 0) = L_JacV.block<3, 12>(0, 6);
        J.block<3, 12>(3, 0) = L_JacW.block<3, 12>(0, 6);
        J.block<3, 12>(6, 0) = R_JacV.block<3, 12>(0, 6);
        J.block<3, 12>(9, 0) = R_JacW.block<3, 12>(0, 6);

        // Solve
        dqj_nm = J.fullPivHouseholderQr().solve(eig_residual);
        // dqj_nm.block<6,1>(0, 0) = J.block<6, 6>(0, 0).fullPivHouseholderQr().solve(eig_residual.block<6, 1>(0, 0));
        // dqj_nm.block<6,1>(6, 0) = J.block<6, 6>(6, 6).fullPivHouseholderQr().solve(eig_residual.block<6, 1>(6, 0));

        // Compute error norms for early termination
        errorf = eig_residual.norm();
        errorX = dqj_nm.norm();

        // Update qj
        eig_qj_op += dqj_nm;

        itr++;
    }

    // Convergence notification
    if (errorf > ef && errorX > ex)
    {
        solconv = false;
        printf("Sol not converged. Errorf: %lg \t ErrorQ: %lg \t Nitr: %d\n", errorf, errorX, itr);
    }
    else
    {
        mju_copy(qj_op, internal_qj_op, 12);
    }

    // Get the angles back in domain -pi to pi
    // May be problematic?
    for (int i = 0; i < 12; i++)
        qj_op[i] = wrap_angles(qj_op[i]);

    mj_deleteData(ik_d);
}

void kondo_12dof_IK(const mjModel *mm, int tsno, int lfsno, int rfsno,
                    double global_torso[], double global_torso_kphi[],
                    double global_leftfoot[], double global_leftfoot_kphi[],
                    double global_rightfoot[], double global_rightfoot_kphi[],
                    double qj_op[],
                    double global_torso_vel[], double global_torso_omega[],
                    double global_leftfoot_vel[], double global_leftfoot_omega[],
                    double global_rightfoot_vel[], double global_rightfoot_omega[],
                    double dqj_op[],
                    int nitrmax, double ef, double ex)
{
    // No processing on the last 10 DOF of Kondo qj_op

    bool solconv = true;
    /*---------Processing Inputs---Convert axis angle to quat------------------------*/
    double global_torso_kaxis[3] = {global_torso_kphi[0], global_torso_kphi[1], global_torso_kphi[2]};
    double global_leftfoot_kaxis[3] = {global_leftfoot_kphi[0], global_leftfoot_kphi[1], global_leftfoot_kphi[2]};
    double global_rightfoot_kaxis[3] = {global_rightfoot_kphi[0], global_rightfoot_kphi[1], global_rightfoot_kphi[2]};
    mju_normalize3(global_torso_kaxis);
    mju_normalize3(global_leftfoot_kaxis);
    mju_normalize3(global_rightfoot_kaxis);
    double global_torso_quat[4], global_leftfoot_quat[4], global_rightfoot_quat[4];
    mju_axisAngle2Quat(global_torso_quat, global_torso_kaxis, global_torso_kphi[3]);
    mju_axisAngle2Quat(global_leftfoot_quat, global_leftfoot_kaxis, global_leftfoot_kphi[3]);
    mju_axisAngle2Quat(global_rightfoot_quat, global_rightfoot_kaxis, global_rightfoot_kphi[3]);

    /*--make mjData instance for IK problem Jacobian computation----------------------*/
    mjData *ik_d = mj_makeData(mm);
    double z3[3] = {0., 0., 0.}, q_eye4[4] = {1, 0, 0, 0}, internal_qj_op[12];
    // Now the floating base is at origin with default orientation--->
    mju_copy3(ik_d->qpos, z3);
    mju_copy4(ik_d->qpos + 3, q_eye4);
    /*
    All calculations in Torso site frame. It has a constant offset w.r.t. the floating base
    left foot pos and orientation,
    right foot and orientation.
    */

    // Create copy of first 12 DOF of the input estimate and update it. Copy back the answer at last if the iterations converge
    mju_copy(internal_qj_op, qj_op, 12);
    Eigen::Map<Eigen::VectorXd> eig_qj_op(internal_qj_op, 12);

    double target_lt_pos[3], target_rt_pos[3], T_target_lt_pos[3], T_target_rt_pos[3];
    double target_lt_quat[4], target_rt_quat[4];
    double R0T[9], RT0[9];
    mju_quat2Mat(R0T, global_torso_quat);
    mju_transpose(RT0, R0T, 3, 3);
    my_mju_relQuat(target_lt_quat, global_leftfoot_quat, global_torso_quat);  // Set desired orientation
    my_mju_relQuat(target_rt_quat, global_rightfoot_quat, global_torso_quat); // Set desied orientation
    mju_sub3(target_lt_pos, global_leftfoot, global_torso);
    mju_mulMatVec(T_target_lt_pos, RT0, target_lt_pos, 3, 3); // Set desired position in Torso frame: RT0(Xl-Xt)
    mju_sub3(target_rt_pos, global_rightfoot, global_torso);
    mju_mulMatVec(T_target_rt_pos, RT0, target_rt_pos, 3, 3); // Set desired position in Torso frame: RT0(Xr-Xt)

    /*--Compute the linear velocity and angular velocity Jacobian matrices-------------*/
    double l_jacV[3 * 28], l_jacW[3 * 28], r_jacV[3 * 28], r_jacW[3 * 28];

    Eigen::Map<Eigen::Matrix<double, 3, 28, Eigen::RowMajor == 1>> L_JacV(l_jacV, 3, 28);
    Eigen::Map<Eigen::Matrix<double, 3, 28, Eigen::RowMajor == 1>> L_JacW(l_jacW, 3, 28);
    Eigen::Map<Eigen::Matrix<double, 3, 28, Eigen::RowMajor == 1>> R_JacV(r_jacV, 3, 28);
    Eigen::Map<Eigen::Matrix<double, 3, 28, Eigen::RowMajor == 1>> R_JacW(r_jacW, 3, 28);
    Eigen::Matrix<double, 12, 12, Eigen::RowMajor == 1> J; // Matrix gradient of residual(qj)

    double residual[12]; // Equation: residual(qj) = 0 is being solved
    Eigen::Map<Eigen::VectorXd> eig_residual(residual, 12);

    double T_curr_lt_pos[3], T_curr_rt_pos[3], curr_lt_quat[4], curr_rt_quat[4];
    Eigen::VectorXd dqj_nm(12); // Output of Newton's method

    int itr = 0;
    double errorf = 1000, errorX = 1000;
    while (errorf > ef && errorX > ex && itr < nitrmax)
    {
        // Set the new qj for revised computations
        mju_copy(ik_d->qpos + 7, internal_qj_op, 12);
        // Compute mj_step1()
        mj_step1(mm, ik_d);
        // Compute Jacobian matrices in Torso frame
        mj_jacSite(mm, ik_d, l_jacV, l_jacW, lfsno);
        mj_jacSite(mm, ik_d, r_jacV, r_jacW, rfsno);

        // Read current position and orientation values (These values are in torso frame only becuase ik_d is set to torso frame)
        my_mju_relQuat(curr_lt_quat, ik_d->sensordata + 10, ik_d->sensordata + 3); // Orientation of left foot w.r.t. torso (Orientation of torso is not going to change!)
        my_mju_relQuat(curr_rt_quat, ik_d->sensordata + 17, ik_d->sensordata + 3); // Orientation of right foot w.r.t. torso (Orientation of torso is not going to change!)
        mju_sub3(T_curr_lt_pos, ik_d->sensordata + 7, ik_d->sensordata);           // T(Xl - Xt) (Xt is not going to change!)
        mju_sub3(T_curr_rt_pos, ik_d->sensordata + 14, ik_d->sensordata);          // T(Xr - Xt) (Xt is not going to change!)

        // Update residual
        mju_sub3(residual, T_target_lt_pos, T_curr_lt_pos);      // compute residual
        mju_subQuat(residual + 3, target_lt_quat, curr_lt_quat); // in tgt space
        mju_sub3(residual + 6, T_target_rt_pos, T_curr_rt_pos);  // compute residual
        mju_subQuat(residual + 9, target_rt_quat, curr_rt_quat); // in tgt space

        // Update big J
        J.block<3, 12>(0, 0) = L_JacV.block<3, 12>(0, 6);
        J.block<3, 12>(3, 0) = L_JacW.block<3, 12>(0, 6);
        J.block<3, 12>(6, 0) = R_JacV.block<3, 12>(0, 6);
        J.block<3, 12>(9, 0) = R_JacW.block<3, 12>(0, 6);

        // Solve
        dqj_nm = J.fullPivHouseholderQr().solve(eig_residual);
        // dqj_nm.block<6,1>(0, 0) = J.block<6, 6>(0, 0).fullPivHouseholderQr().solve(eig_residual.block<6, 1>(0, 0));
        // dqj_nm.block<6,1>(6, 0) = J.block<6, 6>(6, 6).fullPivHouseholderQr().solve(eig_residual.block<6, 1>(6, 0));

        // Compute error norms for early termination
        errorf = eig_residual.norm();
        errorX = dqj_nm.norm();

        // Update qj
        eig_qj_op += dqj_nm;

        itr++;
    }

    // Convergence notification
    if (errorf > ef && errorX > ex)
    {
        solconv = false;
        printf("Sol not converged. Errorf: %lg \t ErrorQ: %lg \t Nitr: %d\n", errorf, errorX, itr);
    }
    else
    {
        mju_copy(qj_op, internal_qj_op, 12);
    }

    // Get the angles back in domain -pi to pi
    // May be problematic?
    for (int i = 0; i < 12; i++)
        qj_op[i] = wrap_angles(qj_op[i]);

    /* Velocity IK */
    // Set the LHS: {v_lt,w_lt,v_rt,w_rt}, i.e., velocities relative to the Torso frame (floating base)
    double cartesian_vw[12];
    mju_sub3(cartesian_vw, global_leftfoot_vel, global_torso_vel);
    mju_sub3(cartesian_vw + 3, global_leftfoot_omega, global_torso_omega);
    mju_sub3(cartesian_vw + 6, global_rightfoot_vel, global_torso_vel);
    mju_sub3(cartesian_vw + 9, global_rightfoot_omega, global_torso_omega);
    Eigen::Map<Eigen::VectorXd> eig_cartesian_vw(cartesian_vw, 12);
    double internal_dqj_op[12];
    Eigen::Map<Eigen::VectorXd> eig_dqj_op(internal_dqj_op, 12);

    /* Update the ik_d and calculate the Jacobian */
    // The floating base position and orientation have been set already (at the time of IK position)
    // Set the converged qj (do not touch the last 10-DOFs)
    mju_copy(ik_d->qpos + 7, qj_op, 12);
    // Compute mj_step1()
    mj_step1(mm, ik_d);
    // Compute Jacobian matrices
    mj_jacSite(mm, ik_d, l_jacV, l_jacW, lfsno);
    mj_jacSite(mm, ik_d, r_jacV, r_jacW, rfsno);
    // Update big J
    J.block<3, 12>(0, 0) = L_JacV.block<3, 12>(0, 6);
    J.block<3, 12>(3, 0) = L_JacW.block<3, 12>(0, 6);
    J.block<3, 12>(6, 0) = R_JacV.block<3, 12>(0, 6);
    J.block<3, 12>(9, 0) = R_JacW.block<3, 12>(0, 6);
    // Solve for dqj_op
    // Solve
    eig_dqj_op = J.fullPivHouseholderQr().solve(eig_cartesian_vw);
    // Write the output joint rates
    mju_copy(dqj_op, internal_dqj_op, 12);

    mj_deleteData(ik_d);
}

void kondo_12dof_CoM_IK_position(const mjModel *mm, int tsno, int lfsno, int rfsno,
                                 double global_com[],
                                 double global_torso[], double global_torso_kphi[],
                                 double global_leftfoot[], double global_leftfoot_kphi[],
                                 double global_rightfoot[], double global_rightfoot_kphi[],
                                 double qj_op[],
                                 int nitrmax, double ef, double ex)
{
    bool solconv = true;
    /*---------Processing Inputs---Convert axis angle to quat------------------------*/
    double global_torso_quat[4], global_leftfoot_quat[4], global_rightfoot_quat[4];
    double global_torso_kaxis[3] = {global_torso_kphi[0], global_torso_kphi[1], global_torso_kphi[2]};
    double global_leftfoot_kaxis[3] = {global_leftfoot_kphi[0], global_leftfoot_kphi[1], global_leftfoot_kphi[2]};
    double global_rightfoot_kaxis[3] = {global_rightfoot_kphi[0], global_rightfoot_kphi[1], global_rightfoot_kphi[2]};
    mju_normalize3(global_torso_kaxis);
    mju_normalize3(global_leftfoot_kaxis);
    mju_normalize3(global_rightfoot_kaxis);
    mju_axisAngle2Quat(global_torso_quat, global_torso_kaxis, global_torso_kphi[3]);
    mju_axisAngle2Quat(global_leftfoot_quat, global_leftfoot_kaxis, global_leftfoot_kphi[3]);
    mju_axisAngle2Quat(global_rightfoot_quat, global_rightfoot_kaxis, global_rightfoot_kphi[3]);

    /*--make mjData instance for IK problem Jacobian computation----------------------*/
    mjData *ik_d = mj_makeData(mm);
    double z3[3] = {0., 0., 0.}, q_eye4[4] = {1, 0, 0, 0}, internal_qj_op[12];
    // Now the floating base is at origin with default orientation--->
    mju_copy3(ik_d->qpos, z3);
    mju_copy4(ik_d->qpos + 3, q_eye4);
    /*
    All calculations in Torso site frame. It has a constant offset w.r.t. the floating base
    left foot pos and orientation, right foot and orientation.
    */

    // Create copy of input estimate and update it. Copy back the answer at last if iterations converge
    mju_copy(internal_qj_op, qj_op, 12);
    Eigen::Map<Eigen::VectorXd> eig_qj_op(internal_qj_op, 12);

    double T_curr_lg_pos[3], T_curr_rg_pos[3], curr_lt_quat[4], curr_rt_quat[4];
    Eigen::VectorXd dqj_nm(12); // Output of Newton's method

    // Control COM-L & COM-R positions and T-L & T-R orientation
    double target_lg_pos[3], target_rg_pos[3], T_target_lg_pos[3], T_target_rg_pos[3];
    double target_lt_quat[4], target_rt_quat[4];
    double R0T[9], RT0[9];
    mju_quat2Mat(R0T, global_torso_quat);
    mju_transpose(RT0, R0T, 3, 3);
    my_mju_relQuat(target_lt_quat, global_leftfoot_quat, global_torso_quat);  // Set desired orientation
    my_mju_relQuat(target_rt_quat, global_rightfoot_quat, global_torso_quat); // Set desied orientation
    mju_sub3(target_lg_pos, global_leftfoot, global_com);
    mju_mulMatVec(T_target_lg_pos, RT0, target_lg_pos, 3, 3); // Set desired position in Torso frame: RT0(Xl-Xg)
    mju_sub3(target_rg_pos, global_rightfoot, global_com);
    mju_mulMatVec(T_target_rg_pos, RT0, target_rg_pos, 3, 3); // Set desired position in Torso frame: RT0(Xr-Xg)

    /*--Compute the linear velocity and angular velocity Jacobian matrices-------------*/
    double l_jacV[3 * 28], l_jacW[3 * 28], r_jacV[3 * 28], r_jacW[3 * 28];
    double g_jacV[3 * 28];
    double lg_jacV[3 * 28], rg_jacV[3 * 28];
    Eigen::Map<Eigen::Matrix<double, 3, 28, Eigen::RowMajor == 1>> LG_JacV(lg_jacV, 3, 28);
    Eigen::Map<Eigen::Matrix<double, 3, 28, Eigen::RowMajor == 1>> L_JacW(l_jacW, 3, 28);
    Eigen::Map<Eigen::Matrix<double, 3, 28, Eigen::RowMajor == 1>> RG_JacV(rg_jacV, 3, 28);
    Eigen::Map<Eigen::Matrix<double, 3, 28, Eigen::RowMajor == 1>> R_JacW(r_jacW, 3, 28);
    Eigen::Matrix<double, 12, 12, Eigen::RowMajor == 1> J; // Matrix gradient of residual(qj)

    double residual[12]; // Equation: residual(qj) = 0 is being solved
    Eigen::Map<Eigen::VectorXd> eig_residual(residual, 12);

    int itr = 0;
    double errorf = 1000, errorX = 1000;
    while (errorf > ef && errorX > ex && itr < 20)
    {
        // Set the new qj for revised computations
        mju_copy(ik_d->qpos + 7, internal_qj_op, 12);
        // Compute mj_step1()
        mj_step1(mm, ik_d);
        // Compute Jacobian matrices in Torso frame
        mj_jacSite(mm, ik_d, l_jacV, l_jacW, lfsno);
        mj_jacSite(mm, ik_d, r_jacV, r_jacW, rfsno);
        mj_jacSubtreeCom(mm, ik_d, g_jacV, 1);
        mju_sub(lg_jacV, l_jacV, g_jacV, 84);
        mju_sub(rg_jacV, r_jacV, g_jacV, 84);

        // Read current position and orientation values (These values are in torso frame only becuase ik_d is set to torso frame)
        my_mju_relQuat(curr_lt_quat, ik_d->sensordata + 10, ik_d->sensordata + 3); // Orientation of left foot w.r.t. torso (Orientation of torso is not going to change!)
        my_mju_relQuat(curr_rt_quat, ik_d->sensordata + 17, ik_d->sensordata + 3); // Orientation of right foot w.r.t. torso (Orientation of torso is not going to change!)
        mju_sub3(T_curr_lg_pos, ik_d->sensordata + 7, ik_d->sensordata + 39);      // T(Xl - Xg)
        mju_sub3(T_curr_rg_pos, ik_d->sensordata + 14, ik_d->sensordata + 39);     // T(Xr - Xg)

        // Update residual
        mju_sub3(residual, T_target_lg_pos, T_curr_lg_pos);      // compute residual
        mju_subQuat(residual + 3, target_lt_quat, curr_lt_quat); // in tgt space
        mju_sub3(residual + 6, T_target_rg_pos, T_curr_rg_pos);  // compute residual
        mju_subQuat(residual + 9, target_rt_quat, curr_rt_quat); // in tgt space

        // Update big J
        J.block<3, 12>(0, 0) = LG_JacV.block<3, 12>(0, 6);
        J.block<3, 12>(3, 0) = L_JacW.block<3, 12>(0, 6);
        J.block<3, 12>(6, 0) = RG_JacV.block<3, 12>(0, 6);
        J.block<3, 12>(9, 0) = R_JacW.block<3, 12>(0, 6);

        // Solve
        dqj_nm = J.fullPivHouseholderQr().solve(eig_residual);

        // Compute error norms for early termination
        errorf = eig_residual.norm();
        errorX = dqj_nm.norm();
        // Update qj
        eig_qj_op += dqj_nm;

        itr++;
    }

    // Convergence notification
    if (errorf > ef && errorX > ex)
    {
        solconv = false;
        printf("Sol not converged. Errorf: %lg \t ErrorQ: %lg \t Nitr: %d\n", errorf, errorX, itr);
    }
    else
    {
        mju_copy(qj_op, internal_qj_op, 12);
    }

    // Get the angles back in domain -pi to pi
    // May be problematic?
    // for (int i = 0; i < 12; i++)
    //     qj_op[i] = wrap_angles(qj_op[i]);

    mj_deleteData(ik_d);
}

void kondo_12dof_CoM_IK(const mjModel *mm, int tsno, int lfsno, int rfsno,
                        double global_com[],
                        double global_torso[], double global_torso_kphi[],
                        double global_leftfoot[], double global_leftfoot_kphi[],
                        double global_rightfoot[], double global_rightfoot_kphi[],
                        double qj_op[],
                        double global_com_vel[],
                        double global_torso_omega[],
                        double global_leftfoot_vel[], double global_leftfoot_omega[],
                        double global_rightfoot_vel[], double global_rightfoot_omega[],
                        double dqj_op[],
                        int nitrmax, double ef, double ex)
{
    bool solconv = true;
    /*---------Processing Inputs---Convert axis angle to quat------------------------*/
    double global_torso_quat[4], global_leftfoot_quat[4], global_rightfoot_quat[4];
    double global_torso_kaxis[3] = {global_torso_kphi[0], global_torso_kphi[1], global_torso_kphi[2]};
    double global_leftfoot_kaxis[3] = {global_leftfoot_kphi[0], global_leftfoot_kphi[1], global_leftfoot_kphi[2]};
    double global_rightfoot_kaxis[3] = {global_rightfoot_kphi[0], global_rightfoot_kphi[1], global_rightfoot_kphi[2]};
    mju_normalize3(global_torso_kaxis);
    mju_normalize3(global_leftfoot_kaxis);
    mju_normalize3(global_rightfoot_kaxis);
    mju_axisAngle2Quat(global_torso_quat, global_torso_kaxis, global_torso_kphi[3]);
    mju_axisAngle2Quat(global_leftfoot_quat, global_leftfoot_kaxis, global_leftfoot_kphi[3]);
    mju_axisAngle2Quat(global_rightfoot_quat, global_rightfoot_kaxis, global_rightfoot_kphi[3]);

    /*--make mjData instance for IK problem Jacobian computation----------------------*/
    mjData *ik_d = mj_makeData(mm);
    double z3[3] = {0., 0., 0.}, q_eye4[4] = {1, 0, 0, 0}, internal_qj_op[12];
    // Now the floating base is at origin with default orientation--->
    mju_copy3(ik_d->qpos, z3);
    mju_copy4(ik_d->qpos + 3, q_eye4);
    /*
    All calculations in Torso site frame. It has a constant offset w.r.t. the floating base
    left foot pos and orientation, right foot and orientation.
    */

    // Create copy of input estimate and update it. Copy back the answer at last if iterations converge
    mju_copy(internal_qj_op, qj_op, 12);
    Eigen::Map<Eigen::VectorXd> eig_qj_op(internal_qj_op, 12);

    double T_curr_lg_pos[3], T_curr_rg_pos[3], curr_lt_quat[4], curr_rt_quat[4];
    Eigen::VectorXd dqj_nm(12); // Output of Newton's method

    // Control COM-L & COM-R positions and T-L & T-R orientation
    double target_lg_pos[3], target_rg_pos[3], T_target_lg_pos[3], T_target_rg_pos[3];
    double target_lt_quat[4], target_rt_quat[4];
    double R0T[9], RT0[9];
    mju_quat2Mat(R0T, global_torso_quat);
    mju_transpose(RT0, R0T, 3, 3);
    my_mju_relQuat(target_lt_quat, global_leftfoot_quat, global_torso_quat);  // Set desired orientation
    my_mju_relQuat(target_rt_quat, global_rightfoot_quat, global_torso_quat); // Set desied orientation
    mju_sub3(target_lg_pos, global_leftfoot, global_com);
    mju_mulMatVec(T_target_lg_pos, RT0, target_lg_pos, 3, 3); // Set desired position in Torso frame: RT0(Xl-Xg)
    mju_sub3(target_rg_pos, global_rightfoot, global_com);
    mju_mulMatVec(T_target_rg_pos, RT0, target_rg_pos, 3, 3); // Set desired position in Torso frame: RT0(Xr-Xg)

    /*--Compute the linear velocity and angular velocity Jacobian matrices-------------*/
    double l_jacV[3 * 28], l_jacW[3 * 28], r_jacV[3 * 28], r_jacW[3 * 28];
    double g_jacV[3 * 28];
    double lg_jacV[3 * 28], rg_jacV[3 * 28];
    Eigen::Map<Eigen::Matrix<double, 3, 28, Eigen::RowMajor == 1>> LG_JacV(lg_jacV, 3, 28);
    Eigen::Map<Eigen::Matrix<double, 3, 28, Eigen::RowMajor == 1>> L_JacW(l_jacW, 3, 28);
    Eigen::Map<Eigen::Matrix<double, 3, 28, Eigen::RowMajor == 1>> RG_JacV(rg_jacV, 3, 28);
    Eigen::Map<Eigen::Matrix<double, 3, 28, Eigen::RowMajor == 1>> R_JacW(r_jacW, 3, 28);
    Eigen::Matrix<double, 12, 12, Eigen::RowMajor == 1> J; // Matrix gradient of residual(qj)

    double residual[12]; // Equation: residual(qj) = 0 is being solved
    Eigen::Map<Eigen::VectorXd> eig_residual(residual, 12);

    int itr = 0;
    double errorf = 1000, errorX = 1000;
    while (errorf > ef && errorX > ex && itr < 20)
    {
        // Set the new qj for revised computations
        mju_copy(ik_d->qpos + 7, internal_qj_op, 12);
        // Compute mj_step1()
        mj_step1(mm, ik_d);
        // Compute Jacobian matrices in Torso frame
        mj_jacSite(mm, ik_d, l_jacV, l_jacW, lfsno);
        mj_jacSite(mm, ik_d, r_jacV, r_jacW, rfsno);
        mj_jacSubtreeCom(mm, ik_d, g_jacV, 1);
        mju_sub(lg_jacV, l_jacV, g_jacV, 84);
        mju_sub(rg_jacV, r_jacV, g_jacV, 84);

        // Read current position and orientation values (These values are in torso frame only becuase ik_d is set to torso frame)
        my_mju_relQuat(curr_lt_quat, ik_d->sensordata + 10, ik_d->sensordata + 3); // Orientation of left foot w.r.t. torso (Orientation of torso is not going to change!)
        my_mju_relQuat(curr_rt_quat, ik_d->sensordata + 17, ik_d->sensordata + 3); // Orientation of right foot w.r.t. torso (Orientation of torso is not going to change!)
        mju_sub3(T_curr_lg_pos, ik_d->sensordata + 7, ik_d->sensordata + 39);      // T(Xl - Xg)
        mju_sub3(T_curr_rg_pos, ik_d->sensordata + 14, ik_d->sensordata + 39);     // T(Xr - Xg)

        // Update residual
        mju_sub3(residual, T_target_lg_pos, T_curr_lg_pos);      // compute residual
        mju_subQuat(residual + 3, target_lt_quat, curr_lt_quat); // in tgt space
        mju_sub3(residual + 6, T_target_rg_pos, T_curr_rg_pos);  // compute residual
        mju_subQuat(residual + 9, target_rt_quat, curr_rt_quat); // in tgt space

        // Update big J
        J.block<3, 12>(0, 0) = LG_JacV.block<3, 12>(0, 6);
        J.block<3, 12>(3, 0) = L_JacW.block<3, 12>(0, 6);
        J.block<3, 12>(6, 0) = RG_JacV.block<3, 12>(0, 6);
        J.block<3, 12>(9, 0) = R_JacW.block<3, 12>(0, 6);

        // Solve
        dqj_nm = J.fullPivHouseholderQr().solve(eig_residual);

        // Compute error norms for early termination
        errorf = eig_residual.norm();
        errorX = dqj_nm.norm();
        // Update qj
        eig_qj_op += dqj_nm;

        itr++;
    }

    // Convergence notification
    if (errorf > ef && errorX > ex)
    {
        solconv = false;
        printf("Sol not converged. Errorf: %lg \t ErrorQ: %lg \t Nitr: %d\n", errorf, errorX, itr);
    }
    else
    {
        mju_copy(qj_op, internal_qj_op, 12);
    }

    /* Velocity COM-IK */
    // Set the LHS: {v_lg,w_lt,v_rg,w_rt}
    double cartesian_vw[12];
    mju_sub3(cartesian_vw, global_leftfoot_vel, global_com_vel);
    mju_sub3(cartesian_vw + 3, global_leftfoot_omega, global_torso_omega);
    mju_sub3(cartesian_vw + 6, global_rightfoot_vel, global_com_vel);
    mju_sub3(cartesian_vw + 9, global_rightfoot_omega, global_torso_omega);
    Eigen::Map<Eigen::VectorXd> eig_cartesian_vw(cartesian_vw, 12);
    double internal_dqj_op[12];
    Eigen::Map<Eigen::VectorXd> eig_dqj_op(internal_dqj_op, 12);

    /* Update the ik_d and calculate the Jacobian */
    // The floating base position and orientation have been set already (at the time of IK position)
    // Set the converged qj
    mju_copy(ik_d->qpos + 7, qj_op, 12);
    // Compute mj_step1()
    mj_step1(mm, ik_d);
    // Compute Jacobian matrices in Torso frame
    mj_jacSite(mm, ik_d, l_jacV, l_jacW, lfsno);
    mj_jacSite(mm, ik_d, r_jacV, r_jacW, rfsno);
    mj_jacSubtreeCom(mm, ik_d, g_jacV, 1);
    mju_sub(lg_jacV, l_jacV, g_jacV, 84);
    mju_sub(rg_jacV, r_jacV, g_jacV, 84);
    // Update big J
    J.block<3, 12>(0, 0) = LG_JacV.block<3, 12>(0, 6);
    J.block<3, 12>(3, 0) = L_JacW.block<3, 12>(0, 6);
    J.block<3, 12>(6, 0) = RG_JacV.block<3, 12>(0, 6);
    J.block<3, 12>(9, 0) = R_JacW.block<3, 12>(0, 6);
    // Solve for dqj_op
    // Solve
    eig_dqj_op = J.fullPivHouseholderQr().solve(eig_cartesian_vw);
    // Write the output joint rates
    mju_copy(dqj_op, internal_dqj_op, 12);

    mj_deleteData(ik_d);
}

double wrap_angles(double angle)
{
    double op;
    op = atan2(sin(angle), cos(angle));
    return op;
}

void deltaPaxispt(const mjModel *mm, mjData *dd, double z, double opzmp[])
{
    // Condim 3: Point contact has only fx and fy friction and N normal force.
    double org[3] = {0, 0, z}; // Origin of the offset plane
    double n[3] = {0, 0, 1};   // Ground normal
    double grf[3] = {0, 0, 0}; // Sum of all the NORMAL contact forces fz
    double grf_loc[6];         // Contact wrench at a point: fx,fy,fz,mx,my,mz
    double r_o_p[3];
    double grf_glo[3];                           // fz in global frame
    double mco[3] = {0, 0, 0}, mco_temp[3];      // Moment of all the NORMAL contact forces about the origin org
    int world_con_id[dd->ncon];                  // No safety check for d->ncon
    int nconw = nconworld(mm, dd, world_con_id); // nconw <= dd->ncon

    // compute the sum of all the contact forces
    // compute the moment of ground contact forces about origin offset to z
    for (int i = 0; i < nconw; i++)
    {
        mj_contactForce(mm, dd, i, grf_loc);                                        // Output in contact frame
        mju_mulMatTVec(grf_glo, dd->contact[world_con_id[i]].frame, grf_loc, 3, 3); // Get fx fy fz in the global frame
        /* It is tricky to get the ground reaction force direction correctly.
        Some times it points correctly out of the ground. But sometimes if doesn't.
        See the notes corresponding to Dec 29-30, 2022 for details.  */
        if (mm->geom_bodyid[dd->contact[world_con_id[i]].geom1])
        {
            // flip all the grf vectors if geom1 does not belong to world body
            mju_scl3(grf_glo, grf_glo, -1);
        }
        mju_addTo3(grf, grf_glo);                               // Sum of all the contact forces
        mju_sub3(r_o_p, dd->contact[world_con_id[i]].pos, org); // Location of contact w.r.t org
        grf_glo[0] = 0;                                         // We have no info of friction in an experiment
        grf_glo[1] = 0;                                         // We have no info of friction in an experiment
        mju_cross(mco_temp, r_o_p, grf_glo);                    // ri x Fi
        mju_addTo3(mco, mco_temp);                              // Sum of all the contact moments abt origin
    }
    // compute the COP-ZMP
    double ndotF;
    ndotF = mju_dot3(n, grf);
    ndotF = 1 / ndotF;
    mju_cross(opzmp, n, mco);
    mju_scl3(opzmp, opzmp, ndotF);
    mju_addTo3(opzmp, org);
}

void deltaCaxispt(const mjModel *mm, mjData *dd, double z, double opzmp[])
{
    // Condim 3: Point contact has only fx and fy friction and N normal force.
    double org[3] = {0, 0, z}; // Origin of the offset plane
    double n[3] = {0, 0, 1};   // Ground normal
    double grf[3] = {0, 0, 0}; // Sum of all the contact forces fx, fy, fz
    double grf_loc[6];         // Contact wrench at a point: fx,fy,fz,mx,my,mz
    double r_o_p[3];
    double grf_glo[3];                           // fx, fy, fz in global frame
    double mco[3] = {0, 0, 0}, mco_temp[3];      // Moment of all the contact forces about the origin org
    int world_con_id[dd->ncon];                  // No safety check for d->ncon
    int nconw = nconworld(mm, dd, world_con_id); // nconw <= dd->ncon

    // compute the sum of all the contact forces
    // compute the moment of ground contact forces about origin offset to z
    for (int i = 0; i < nconw; i++)
    {
        mj_contactForce(mm, dd, i, grf_loc);                                        // Output in contact frame
        mju_mulMatTVec(grf_glo, dd->contact[world_con_id[i]].frame, grf_loc, 3, 3); // Get fx fy fz in the global frame
        /* It is tricky to get the ground reaction force direction correctly.
        Some times it points correctly out of the ground. But sometimes if doesn't.
        See the notes corresponding to Dec 29-30, 2022 for details.  */
        if (mm->geom_bodyid[dd->contact[world_con_id[i]].geom1])
        {
            // flip all the grf vectors if geom1 does not belong to world body
            mju_scl3(grf_glo, grf_glo, -1);
        }
        mju_addTo3(grf, grf_glo);                               // Sum of all the contact forces
        mju_sub3(r_o_p, dd->contact[world_con_id[i]].pos, org); // Location of contact w.r.t org
        mju_cross(mco_temp, r_o_p, grf_glo);                    // ri x Fi
        mju_addTo3(mco, mco_temp);                              // Sum of all the contact moments abt origin
    }
    // compute the COP-ZMP
    double ndotF;
    ndotF = mju_dot3(n, grf);
    ndotF = 1 / ndotF;
    mju_cross(opzmp, n, mco);
    mju_scl3(opzmp, opzmp, ndotF);
    mju_addTo3(opzmp, org);
}

double my_mju_powerbalance(const mjModel *mm, mjData *dd)
{
    // Missing joint dry friction energy dissipation
    // Allocated for this function: d->userdata[10 to 19]
    mj_forward(mm, dd);                                                                 // May reduce this overhead
    double actuator_power = mju_dot(dd->actuator_force, dd->actuator_velocity, mm->nu); // Instantaneous actuator power input
    dd->userdata[12] = dd->energy[0] + dd->energy[1];                                   // Energy now: TE = PE + KE
    double de = (dd->userdata[12] - dd->userdata[10]) / (2 * mm->opt.timestep);         // Energy dissipation or addition rate
    dd->userdata[11] = dd->userdata[12];                                                // Update the old value of energy
    dd->userdata[10] = dd->userdata[11];
    double residual = (de - actuator_power) / (12 * 0.5 * 3); // Error (Watt)/(Watt Normalizer: 12 motors x 0.5 Nm nominal torque x 3 rad/s qdot)
    return residual;
}

/*-------Draw Decorative Geometries----------*/

void drawEllipsoid(const mjModel *mm, mjData *dd, mjtNum pt[], mjtNum semi_axes[], mjtNum *quat, float rgba[], mjvScene *scene, const mjvOption *opt)
{
    // add a decorative geometry
    mjvGeom *mygeom;
    mjtNum myrot3x3[9] = {1., 0., 0., 0., 1., 0., 0., 0., 1.};
    mju_quat2Mat(myrot3x3, quat);

    // one more geom to render
    scene->ngeom = scene->ngeom + 1;
    // mygeom now points to the last location in geoms buffer
    mygeom = scene->geoms + scene->ngeom - 1;
    // decor geom
    mygeom->objtype = mjOBJ_UNKNOWN;
    mygeom->objid = -1;
    mygeom->category = mjCAT_DECOR;
    mygeom->segid = scene->ngeom;

    // Add it to the scene
    mjv_initGeom(mygeom, mjGEOM_ELLIPSOID, semi_axes, pt, myrot3x3, rgba);
    mjv_addGeoms(mm, dd, opt, NULL, mjCAT_DECOR, scene);
}

void drawPointSph(const mjModel *mm, mjData *dd, mjtNum pt[3], mjtNum radius, float rgba[], mjvScene *scene, const mjvOption *opt)
{
    // add a decorative geometry
    mjvGeom *mygeom;
    mjtNum sphsize[3] = {radius, 0, 0};
    mjtNum sphpos[3] = {0., 0., 0.};
    mjtNum myrot3x3[9] = {1., 0., 0., 0., 1., 0., 0., 0., 1.};

    mju_copy3(sphpos, pt);

    // one more geom to render
    scene->ngeom = scene->ngeom + 1;
    // mygeom now points to the last location in geoms buffer
    mygeom = scene->geoms + scene->ngeom - 1;
    // decor geom
    mygeom->objtype = mjOBJ_UNKNOWN;
    mygeom->objid = -1;
    mygeom->category = mjCAT_DECOR;
    mygeom->segid = scene->ngeom;

    // Add it to the scene
    mjv_initGeom(mygeom, mjGEOM_SPHERE, sphsize, sphpos, myrot3x3, rgba);
    mjv_addGeoms(mm, dd, opt, NULL, mjCAT_DECOR, scene);
}

void drawPointBox(const mjModel *mm, mjData *dd, mjtNum pt[3], mjtNum dim, float rgba[], mjvScene *scene, const mjvOption *opt)
{
    // add a decorative geometry
    mjvGeom *mygeom;
    mjtNum boxsize[3] = {dim, dim, dim};
    mjtNum sphpos[3] = {0., 0., 0.};
    mjtNum myrot3x3[9] = {1., 0., 0., 0., 1., 0., 0., 0., 1.};

    mju_copy3(sphpos, pt);

    // one more geom to render
    scene->ngeom = scene->ngeom + 1;
    // mygeom now points to the last location in geoms buffer
    mygeom = scene->geoms + scene->ngeom - 1;
    // decor geom
    mygeom->objtype = mjOBJ_UNKNOWN;
    mygeom->objid = -1;
    mygeom->category = mjCAT_DECOR;
    mygeom->segid = scene->ngeom;

    // Add it to the scene
    mjv_initGeom(mygeom, mjGEOM_BOX, boxsize, sphpos, myrot3x3, rgba);
    mjv_addGeoms(mm, dd, opt, NULL, mjCAT_DECOR, scene);
}

void drawPush(mjModel *mm, mjData *dd, mjvScene *scene, const mjvOption *opt, int body_id)
{
    // Force display
    mjvGeom *mygeom;
    mjtNum linesize[3] = {0., 0., -1};
    mjtNum linepos[3] = {0., 0., 0.};
    mjtNum diff[3];
    mjtNum myquat[4] = {1, 0, 0, 0};
    mjtNum myrot3x3[9] = {1., 0., 0., 0., 1., 0., 0., 0., 1.};
    float linergba[4] = {0., 0.9, 0., 0.9}; // Green for force

    mju_copy3(linepos, dd->xipos + 3 * body_id);                                // Arrow start: linepos == body_com
    mju_copy3(diff, dd->xfrc_applied + 6 * body_id);                            // Arrow end: at body_com + xfrc_applied
    linesize[2] = mju_norm3(diff) * 20. * mm->vis.map.force / mm->stat.meanmass; // Arrow length
    linesize[0] = mm->vis.scale.forcewidth * mm->stat.meansize;                 // Arrow rad (minor radius)
    linesize[1] = linesize[0];                                                  // Arrow rad (major radius)
    // Show arrow only when there is non-zero pushing force
    if (linesize[2] != 0)
    {
        // set mat to minimal rotation aligning b-a with z axis
        mju_quatZ2Vec(myquat, diff);
        mju_quat2Mat(myrot3x3, myquat);

        scene->ngeom = scene->ngeom + 1;          // one more geom to render
        mygeom = scene->geoms + scene->ngeom - 1; // mygeom now points to the last location in geoms buffer
        mygeom->objtype = mjOBJ_UNKNOWN;          // decor geom
        mygeom->objid = -1;                       //
        mygeom->category = mjCAT_DECOR;           // Decorative geometry
        mygeom->segid = scene->ngeom;

        mjv_initGeom(mygeom, mjGEOM_ARROW, linesize, linepos, myrot3x3, linergba);
        mjv_addGeoms(mm, dd, opt, NULL, mjCAT_DECOR, scene); // Add it to the scene
    }

    // Moment display
    mjvGeom *mygeom2;
    mjtNum linesize2[3] = {0., 0., -1};
    mjtNum linepos2[3] = {0., 0., 0.};
    mjtNum diff2[3];
    mjtNum myquat2[4] = {1, 0, 0, 0};
    mjtNum myrot3x3_2[9] = {1., 0., 0., 0., 1., 0., 0., 0., 1.};
    float linergba2[4] = {0.9, 0., 0., 0.9}; // Red for moment

    mju_copy3(linepos2, dd->xipos + 3 * body_id);                                 // Arrow start: linepos == body_com
    mju_copy3(diff2, dd->xfrc_applied + 6 * body_id + 3);                         // Arrow end: at body_com + xfrc_applied
    linesize2[2] = mju_norm3(diff2) * 6. * mm->vis.map.force / mm->stat.meanmass; // Arrow length
    linesize2[0] = mm->vis.scale.forcewidth * mm->stat.meansize;                  // Arrow rad (minor radius)
    linesize2[1] = linesize2[0];                                                  // Arrow rad (major radius)
    // Show arrow only when there is non-zero external pushing moment
    if (linesize2[2] != 0)
    {
        // set mat to minimal rotation aligning b-a with z axis
        mju_quatZ2Vec(myquat2, diff2);
        mju_quat2Mat(myrot3x3_2, myquat2);

        scene->ngeom = scene->ngeom + 1;           // one more geom to render
        mygeom2 = scene->geoms + scene->ngeom - 1; // mygeom now points to the last location in geoms buffer
        mygeom2->objtype = mjOBJ_UNKNOWN;          // decor geom
        mygeom2->objid = -1;                       //
        mygeom2->category = mjCAT_DECOR;           // Decorative geometry
        mygeom2->segid = scene->ngeom;

        mjv_initGeom(mygeom2, mjGEOM_ARROW, linesize2, linepos2, myrot3x3_2, linergba2);
        mjv_addGeoms(mm, dd, opt, NULL, mjCAT_DECOR, scene); // Add it to the scene
    }
}

void drawWorldFrame(mjModel *mm, mjData *dd, mjvScene *scene, const mjvOption *opt)
{
    // add a decorative geometry
    mjvGeom *mygeom1, *mygeom2, *mygeom3;
    mjtNum scl = mm->stat.meansize;
    mjtNum arrowsize[3] = {0.003, 0.003, 1};
    mjtNum arrowpos[3] = {0., 0., 0.};
    mjtNum myrot3x3[9] = {1., 0., 0., 0., 1., 0., 0., 0., 1.};
    // Red X -- Green Y -- BLue Z
    float arrowrgbaX[4] = {0.9, 0., 0., 0.8};
    float arrowrgbaY[4] = {0, 0.9, 0., 0.8};
    float arrowrgbaZ[4] = {0., 0., 0.9, 0.8};
    mjtNum arrowscale = scl * 2;

    // Set Arrow length:
    arrowsize[2] = arrowscale;

    // Set Arrow direction:
    mjtNum diffX[3] = {1, 0, 0};
    mjtNum diffY[3] = {0, 1, 0};
    mjtNum diffZ[3] = {0, 0, 1};
    mjtNum myquatX[4] = {1, 0, 0, 0};
    mjtNum myquatY[4] = {1, 0, 0, 0};
    mjtNum myquatZ[4] = {1, 0, 0, 0};

    // set mat to minimal rotation aligning b-a with z axis
    mju_quatZ2Vec(myquatX, diffX);
    mju_quat2Mat(myrot3x3, myquatX);

    // one more geom to render
    scene->ngeom = scene->ngeom + 1;
    // mygeom now points to the last location in geoms buffer
    mygeom1 = scene->geoms + scene->ngeom - 1;
    // decor geom
    mygeom1->objtype = mjOBJ_UNKNOWN;
    mygeom1->objid = -1;
    mygeom1->category = mjCAT_DECOR;
    mygeom1->segid = scene->ngeom;

    // Add it to the scene
    mjv_initGeom(mygeom1, mjGEOM_ARROW, arrowsize, arrowpos, myrot3x3, arrowrgbaX);
    // mjv_addGeoms(mm, dd, opt, NULL, mjCAT_DECOR, scene);
    /*--------------------------------------------------------------------------------*/
    // set mat to minimal rotation aligning b-a with z axis
    mju_quatZ2Vec(myquatY, diffY);
    mju_quat2Mat(myrot3x3, myquatY);

    // one more geom to render
    scene->ngeom = scene->ngeom + 1;
    // mygeom now points to the last location in geoms buffer
    mygeom2 = scene->geoms + scene->ngeom - 1;
    // decor geom
    mygeom2->objtype = mjOBJ_UNKNOWN;
    mygeom2->objid = -1;
    mygeom2->category = mjCAT_DECOR;
    mygeom2->segid = scene->ngeom;

    // Add it to the scene
    mjv_initGeom(mygeom2, mjGEOM_ARROW, arrowsize, arrowpos, myrot3x3, arrowrgbaY);
    // mjv_addGeoms(mm, dd, opt, NULL, mjCAT_DECOR, scene);
    /*--------------------------------------------------------------------------------*/
    // set mat to minimal rotation aligning b-a with z axis
    mju_quatZ2Vec(myquatZ, diffZ);
    mju_quat2Mat(myrot3x3, myquatZ);

    // one more geom to render
    scene->ngeom = scene->ngeom + 1;
    // mygeom now points to the last location in geoms buffer
    mygeom3 = scene->geoms + scene->ngeom - 1;
    // decor geom
    mygeom3->objtype = mjOBJ_UNKNOWN;
    mygeom3->objid = -1;
    mygeom3->category = mjCAT_DECOR;
    mygeom3->segid = scene->ngeom;

    // Add it to the scene
    mjv_initGeom(mygeom3, mjGEOM_ARROW, arrowsize, arrowpos, myrot3x3, arrowrgbaZ);
    mjv_addGeoms(mm, dd, opt, NULL, mjCAT_DECOR, scene);
    /*--------------------------------------------------------------------------------*/
}

void drawCOM(mjModel *mm, mjData *dd, mjvScene *scene, const mjvOption *opt)
{
    // add a decorative geometry
    mjvGeom *mygeom;
    mjtNum sphsize[3] = {0.005, 0, 0};
    mjtNum sphpos[3] = {0., 0., 0.};
    mjtNum myrot3x3[9] = {1., 0., 0., 0., 1., 0., 0., 0., 1.};
    float sphrgba[4] = {0.9, 0., 0., 0.9}; // Red

    mju_copy3(sphpos, dd->subtree_com + 1 * 3);

    // one more geom to render
    scene->ngeom = scene->ngeom + 1;
    // mygeom now points to the last location in geoms buffer
    mygeom = scene->geoms + scene->ngeom - 1;
    // decor geom
    mygeom->objtype = mjOBJ_UNKNOWN;
    mygeom->objid = -1;
    mygeom->category = mjCAT_DECOR;
    mygeom->segid = scene->ngeom;

    // Add it to the scene
    mjv_initGeom(mygeom, mjGEOM_SPHERE, sphsize, sphpos, myrot3x3, sphrgba);
    mjv_addGeoms(mm, dd, opt, NULL, mjCAT_DECOR, scene);
}

void drawCOMproj(mjModel *mm, mjData *dd, mjvScene *scene, const mjvOption *opt)
{

    //----------------------------------------------------------------------
    // add a line decorative geometry
    mjvGeom *mygeom2;
    mjtNum linesize[3] = {0., 0., -1};
    mjtNum linepos[3] = {0., 0., 0.};
    mjtNum com_proj[3];
    mjtNum diff[3];
    mjtNum myquat[4] = {1, 0, 0, 0};
    mjtNum myrot3x3_2[9] = {1., 0., 0., 0., 1., 0., 0., 0., 1.};
    // Red
    float linergba[4] = {0.9, 0., 0., 0.9};

    // line starts at linepos
    mju_copy3(linepos, dd->subtree_com + 1 * 3);
    com_proj[0] = *(dd->subtree_com + 3);
    com_proj[1] = *(dd->subtree_com + 3 + 1);
    // Line end: com_proj, line start: com
    mju_sub3(diff, com_proj, dd->subtree_com + 1 * 3);
    // line length = height of COM from ground
    linesize[2] = mju_norm3(diff);

    // set mat to minimal rotation aligning b-a with z axis
    mju_quatZ2Vec(myquat, diff);
    mju_quat2Mat(myrot3x3_2, myquat);

    // one more geom to render
    scene->ngeom = scene->ngeom + 1;
    // mygeom now points to the last location in geoms buffer
    mygeom2 = scene->geoms + scene->ngeom - 1;
    // decor geom
    mygeom2->objtype = mjOBJ_UNKNOWN;
    mygeom2->objid = -1;
    mygeom2->category = mjCAT_DECOR;
    mygeom2->segid = scene->ngeom;

    // Add it to the scene
    mjv_initGeom(mygeom2, mjGEOM_LINE, linesize, linepos, myrot3x3_2, linergba);
    mjv_addGeoms(mm, dd, opt, NULL, mjCAT_DECOR, scene);
}

void drawCOPcongrp(mjModel *mm, mjData *dd, int grpno, ContactDataAnalysis *cda, mjvScene *scene, const mjvOption *opt)
{
    // Draw COP of contact group number grpno.
    // grpno starts from 0 index
    // add a decorative geometry
    mjvGeom *mygeom;
    mjtNum sphsize[3] = {0.005, 0.005, 0.005};
    // COP position
    mjtNum sphpos[3] = {0., 0., 0.};
    mjtNum myrot3x3[9] = {1., 0., 0., 0., 1., 0., 0., 0., 1.};
    float sphrgba[4] = {0., 0., 0.9, 0.9};

    mjtNum rcen_dummy[3] = {0, 0, 0};
    mjtNum totCF_dummy[3] = {0, 0, 0};

    if (cda->ngroups > (grpno))
    {
        // Find values of rcop, rcen and total contact force
        contactgroupdata(mm, dd, grpno, sphpos, rcen_dummy, totCF_dummy);
        // one more geom to render
        scene->ngeom = scene->ngeom + 1;
        // mygeom now points to the last location in geoms buffer
        mygeom = scene->geoms + scene->ngeom - 1;
        // decor geom
        mygeom->objtype = mjOBJ_UNKNOWN;
        mygeom->objid = -1;
        mygeom->category = mjCAT_DECOR;
        mygeom->segid = scene->ngeom;

        // Add it to the scene
        mjv_initGeom(mygeom, mjGEOM_BOX, sphsize, sphpos, myrot3x3, sphrgba);
        mjv_addGeoms(mm, dd, opt, NULL, mjCAT_DECOR, scene);
    }
}

void drawCENcongrp(mjModel *mm, mjData *dd, int grpno, mjvScene *scene, const mjvOption *opt)
{
    // Draw COP of contact group number grpno.
    // grpno starts from 0 index
    // add a decorative geometry
    mjvGeom *mygeom;
    mjtNum sphsize[3] = {0.003, 0, 0};
    // COP position
    mjtNum sphpos[3] = {0., 0., 0.};
    mjtNum myrot3x3[9] = {1., 0., 0., 0., 1., 0., 0., 0., 1.};
    float sphrgba[4] = {0., 0.5, 0.5, 0.9};

    mjtNum rcop[3] = {0, 0, 0};
    mjtNum totCF[3] = {0, 0, 0};
    // Find values of rcop, rcen and total contact force
    contactgroupdata(mm, dd, grpno, rcop, sphpos, totCF);

    // one more geom to render
    scene->ngeom = scene->ngeom + 1;
    // mygeom now points to the last location in geoms buffer
    mygeom = scene->geoms + scene->ngeom - 1;
    // decor geom
    mygeom->objtype = mjOBJ_UNKNOWN;
    mygeom->objid = -1;
    mygeom->category = mjCAT_DECOR;
    mygeom->segid = scene->ngeom;

    // Add it to the scene
    mjv_initGeom(mygeom, mjGEOM_SPHERE, sphsize, sphpos, myrot3x3, sphrgba);
    mjv_addGeoms(mm, dd, opt, NULL, mjCAT_DECOR, scene);
}

void drawLineSeg(mjModel *mm, mjData *dd, mjtNum p1[], mjtNum p2[], float rgba[], mjvScene *scene, const mjvOption *opt)
{
    mjvGeom *mylinegeom;
    mjtNum linesize[3] = {0., 0., -1};
    mjtNum linepos[3] = {0., 0., 0.};
    mjtNum com_proj[3];
    mjtNum diff[3];
    mjtNum myquat[4] = {1, 0, 0, 0};
    mjtNum myrot3x3_2[9] = {1., 0., 0., 0., 1., 0., 0., 0., 1.};

    // Line start: at ith point
    mju_copy3(linepos, p1);
    // Line end: at i+1st point
    mju_sub3(diff, p2, p1);
    // line length
    linesize[2] = mju_norm3(diff);

    // set mat to minimal rotation aligning b-a with z axis
    mju_quatZ2Vec(myquat, diff);
    mju_quat2Mat(myrot3x3_2, myquat);

    // one more geom to render
    scene->ngeom = scene->ngeom + 1;
    // mygeom now points to the last location in geoms buffer
    mylinegeom = scene->geoms + scene->ngeom - 1;
    // decor geom
    mylinegeom->objtype = mjOBJ_UNKNOWN;
    mylinegeom->objid = -1;
    mylinegeom->category = mjCAT_DECOR;
    mylinegeom->segid = scene->ngeom;
    // Add it to the scene
    mjv_initGeom(mylinegeom, mjGEOM_LINE, linesize, linepos, myrot3x3_2, rgba);

    mjv_addGeoms(mm, dd, opt, NULL, mjCAT_DECOR, scene);
}

void drawPlaneRect(mjModel *mm, mjData *dd, mjtNum pc[], mjtNum normal[], mjtNum a, mjtNum b, float rgba[], mjvScene *scene, const mjvOption *opt)
{
    mjvGeom *myplanegeom;
    mjtNum planesize[3] = {a, b, 0.01}; // Half widht, half length and grid spacing for rendering.
    mjtNum myrot3x3[9] = {1., 0., 0., 0., 1., 0., 0., 0., 1.};
    mjtNum myquat[4] = {1, 0, 0, 0};
    // Set the geometry frame such that the z axis aligns to the normal vector
    mju_quatZ2Vec(myquat, normal);
    mju_quat2Mat(myrot3x3, myquat);

    // one more geom to render
    scene->ngeom = scene->ngeom + 1;
    // mygeom now points to the last location in geoms buffer
    myplanegeom = scene->geoms + scene->ngeom - 1;
    // decor geom
    myplanegeom->objtype = mjOBJ_UNKNOWN;
    myplanegeom->objid = -1;
    myplanegeom->category = mjCAT_DECOR;
    myplanegeom->segid = scene->ngeom;
    // Initialize the geometry
    mjv_initGeom(myplanegeom, mjGEOM_PLANE, planesize, pc, myrot3x3, rgba);
}

void drawPolygon(const mjModel *mm, mjData *dd, float rgba[], mjvScene *scene, const mjvOption *opt, double pts[], int n)
{
    // Reminder: Size of pts = (n+1) x 3

    mjvGeom *vertices[n], *edges[n];
    //--------Plot Vertices-------------------------------------------------
    // Set details
    mjtNum sphsize[3] = {0.005, 0, 0};
    mjtNum myrot3x3[9] = {1., 0., 0., 0., 1., 0., 0., 0., 1.};
    for (int i = 0; i < n; i++)
    {
        // one more geom to render
        scene->ngeom = scene->ngeom + 1;
        // mygeom now points to the last location in geoms buffer
        vertices[i] = scene->geoms + scene->ngeom - 1;
        // decor geom
        vertices[i]->objtype = mjOBJ_UNKNOWN;
        vertices[i]->objid = -1;
        vertices[i]->category = mjCAT_DECOR;
        vertices[i]->segid = scene->ngeom;
        // Initialize the geometry
        mjv_initGeom(vertices[i], mjGEOM_SPHERE, sphsize, pts + 3 * i, myrot3x3, rgba);
    }

    //--------Plot Edges-------------------------------------------------
    mjtNum linesize[3] = {0., 0., -1};
    mjtNum linepos[3] = {0., 0., 0.};
    mjtNum com_proj[3];
    mjtNum diff[3];
    mjtNum myquat[4] = {1, 0, 0, 0};
    mjtNum myrot3x3_2[9] = {1., 0., 0., 0., 1., 0., 0., 0., 1.};
    for (int i = 0; i < n; i++)
    {

        // Line start: at ith point
        mju_copy3(linepos, pts + 3 * i);
        // Line end: at i+1st point
        mju_sub3(diff, pts + 3 * (i + 1), pts + 3 * i);
        // line length
        linesize[2] = mju_norm3(diff);

        // set mat to minimal rotation aligning b-a with z axis
        mju_quatZ2Vec(myquat, diff);
        mju_quat2Mat(myrot3x3_2, myquat);

        // one more geom to render
        scene->ngeom = scene->ngeom + 1;
        // mygeom now points to the last location in geoms buffer
        edges[i] = scene->geoms + scene->ngeom - 1;
        // decor geom
        edges[i]->objtype = mjOBJ_UNKNOWN;
        edges[i]->objid = -1;
        edges[i]->category = mjCAT_DECOR;
        edges[i]->segid = scene->ngeom;
        // Add it to the scene
        mjv_initGeom(edges[i], mjGEOM_LINE, linesize, linepos, myrot3x3_2, rgba);
    }

    mjv_addGeoms(mm, dd, opt, NULL, mjCAT_DECOR, scene);
}

void drawLineFootCOPs(mjModel *mm, mjData *dd, int grpno1, int grpno2, mjvScene *scene, const mjvOption *opt)
{
    mjtNum cop1[3], cop2[3], cen1[3], cen2[3], tcf1[3], tcf2[3];
    float colour[4] = {0., 0.0, 0.9, 0.9}; //  Blue
    contactgroupdata(mm, dd, grpno1, cop1, cen1, tcf1);
    contactgroupdata(mm, dd, grpno2, cop2, cen2, tcf2);
    drawLineSeg(mm, dd, cop1, cop2, colour, scene, opt);
}

void drawGenCOP(mjModel *mm, mjData *dd, ContactDataAnalysis *cda, mjvScene *scene, const mjvOption *opt)
{

    mjvGeom *mycopgeom;
    //--------Plot the Gen COP----------------------------------------
    // Set details
    mjtNum boxpos[3] = {0, 0, 0}; // gen COP position
    mjtNum boxsize[3] = {0.005, 0.005, 0.005};
    mjtNum myrot3x3[9] = {1., 0., 0., 0., 1., 0., 0., 0., 1.};
    float boxrgba[4] = {0., 0., 0.9, 0.9}; // Blue

    mju_copy3(boxpos, cda->gcop); // gen COP position

    // one more geom to render
    scene->ngeom = scene->ngeom + 1;
    // mygeom now points to the last location in geoms buffer
    mycopgeom = scene->geoms + scene->ngeom - 1;
    // decor geom
    mycopgeom->objtype = mjOBJ_UNKNOWN;
    mycopgeom->objid = -1;
    mycopgeom->category = mjCAT_DECOR;
    mycopgeom->segid = scene->ngeom;
    // Initialize the geometry
    mjv_initGeom(mycopgeom, mjGEOM_BOX, boxsize, boxpos, myrot3x3, boxrgba);
}

void drawdeltaPaxis(mjModel *mm, mjData *dd, double z1, double z2, mjvScene *scene, const mjvOption *opt)
{
    // GREEN
    // genCOP by default belongs to this. and drawn in scene with this logic. not genZMP logic.
    // add a line decorative geometry
    mjvGeom *mygeom2;
    mjtNum linesize[3] = {0., 0., -1};
    mjtNum linepos[3], linepos2[3];
    mjtNum com_proj[3];
    mjtNum diff[3];
    mjtNum myquat[4] = {1, 0, 0, 0};
    mjtNum myrot3x3_2[9] = {1., 0., 0., 0., 1., 0., 0., 0., 1.};
    float linergba[4] = {0., 0.9, 0., 0.9}; // Green

    // line starts at linepos, i.e., ZMP-1
    deltaPaxispt(mm, dd, z1, linepos);
    // Line end: ZMP-2, line start: ZMP-1
    deltaPaxispt(mm, dd, z2, linepos2);
    mju_sub3(diff, linepos2, linepos);
    // line length = distance between ZMP-1 and ZMP-2
    linesize[2] = mju_norm3(diff);

    // set mat to minimal rotation aligning b-a with z axis
    mju_quatZ2Vec(myquat, diff);
    mju_quat2Mat(myrot3x3_2, myquat);

    // one more geom to render
    scene->ngeom = scene->ngeom + 1;
    // mygeom now points to the last location in geoms buffer
    mygeom2 = scene->geoms + scene->ngeom - 1;
    // decor geom
    mygeom2->objtype = mjOBJ_UNKNOWN;
    mygeom2->objid = -1;
    mygeom2->category = mjCAT_DECOR;
    mygeom2->segid = scene->ngeom;

    // Add it to the scene
    mjv_initGeom(mygeom2, mjGEOM_LINE, linesize, linepos, myrot3x3_2, linergba);
    mjv_addGeoms(mm, dd, opt, NULL, mjCAT_DECOR, scene);
}

void drawdeltaCaxis(mjModel *mm, mjData *dd, double z1, double z2, mjvScene *scene, const mjvOption *opt)
{
    // YELLOW
    // add a line decorative geometry
    mjvGeom *mygeom2;
    mjtNum linesize[3] = {0., 0., -1};
    mjtNum linepos[3], linepos2[3];
    mjtNum com_proj[3];
    mjtNum diff[3];
    mjtNum myquat[4] = {1, 0, 0, 0};
    mjtNum myrot3x3_2[9] = {1., 0., 0., 0., 1., 0., 0., 0., 1.};
    float linergba[4] = {0.9, 0.9, 0., 0.9}; // Yellow

    // line starts at linepos, i.e., ZMP-1
    deltaCaxispt(mm, dd, z1, linepos);
    // Line end: ZMP-2, line start: ZMP-1
    deltaCaxispt(mm, dd, z2, linepos2);
    mju_sub3(diff, linepos2, linepos);
    // line length = distance between ZMP-1 and ZMP-2
    linesize[2] = mju_norm3(diff);

    // set mat to minimal rotation aligning b-a with z axis
    mju_quatZ2Vec(myquat, diff);
    mju_quat2Mat(myrot3x3_2, myquat);

    // one more geom to render
    scene->ngeom = scene->ngeom + 1;
    // mygeom now points to the last location in geoms buffer
    mygeom2 = scene->geoms + scene->ngeom - 1;
    // decor geom
    mygeom2->objtype = mjOBJ_UNKNOWN;
    mygeom2->objid = -1;
    mygeom2->category = mjCAT_DECOR;
    mygeom2->segid = scene->ngeom;

    // Add it to the scene
    mjv_initGeom(mygeom2, mjGEOM_LINE, linesize, linepos, myrot3x3_2, linergba);
    mjv_addGeoms(mm, dd, opt, NULL, mjCAT_DECOR, scene);
}

void drawFootRotationAxis(const mjModel *mm, mjData *dd, mjvScene *scene, const mjvOption *opt)
{
    double torso_gyr[3], lf_gyr[3], rf_gyr[3];
    double torso_mat[9], lf_mat[9], rf_mat[9];

    mju_mulMatVec(torso_gyr, dd->xmat + 1 * 9, dd->sensordata + 24, 3, 3); // Torso Omega in global frame, body id 1
    mju_copy3(lf_gyr, dd->sensordata + 30);                                // Left foot Omega in body frame
    mju_mulMatVec(lf_gyr, dd->xmat + 7 * 9, dd->sensordata + 24, 3, 3);    // Left foot Omega in global frame, body id 7
    mju_copy3(rf_gyr, dd->sensordata + 36);                                // Right foot Omega in body frame
    mju_mulMatVec(rf_gyr, dd->xmat + 13 * 9, dd->sensordata + 24, 3, 3);   // Right foot Omega in global frame, body id 13

    double cp1[3], cp2[3];           // Contact points of the rotation edge
    double tilt_axis[3] = {0, 0, 0}; // Tilt axis

    if (dd->ncon == 2)
    {

        mju_copy3(cp1, dd->contact[0].pos); // Global position of contact point 1
        mju_copy3(cp2, dd->contact[1].pos); // Global position of contact point 2
        mju_sub3(tilt_axis, cp1, cp2);      // tilt_axis = cp1 - cp2
        // mju_normalize3(tilt_axis);
        // Compare rotations of torso, left foot, right foot, and the tilt axis
    }

    //--------Plot the left foot site Omega arrow----------------------------
    mjvGeom *omegaL;
    mjtNum arrowsize[3] = {0.005, 0.005, -1};
    mjtNum arrowpos[3] = {0., 0., 0.};
    mjtNum diff[3];
    mjtNum myquat[4] = {1, 0, 0, 0};
    mjtNum myrot3x3_2[9] = {1., 0., 0., 0., 1., 0., 0., 0., 1.};
    float arrowrgba[4] = {1, 1, 1, 0.9}; // White

    // Line start: at site left foot / torso
    mju_copy3(arrowpos, dd->sensordata /* + 7 */);
    // Line end: w of left foot
    mju_copy3(diff, lf_gyr);
    // line length
    arrowsize[2] = mju_norm3(diff);

    // set mat to minimal rotation aligning b-a with z axis
    mju_quatZ2Vec(myquat, diff);
    mju_quat2Mat(myrot3x3_2, myquat);

    // one more geom to render
    scene->ngeom = scene->ngeom + 1;
    // mygeom now points to the last location in geoms buffer
    omegaL = scene->geoms + scene->ngeom - 1;
    // decor geom
    omegaL->objtype = mjOBJ_UNKNOWN;
    omegaL->objid = -1;
    omegaL->category = mjCAT_DECOR;
    omegaL->segid = scene->ngeom;
    // Add it to the scene
    mjv_initGeom(omegaL, mjGEOM_ARROW, arrowsize, arrowpos, myrot3x3_2, arrowrgba);

    //--------Plot the right foot site Omega arrow----------------------------
    mjvGeom *omegaR;

    // Line start: at site right foot/ torso
    mju_copy3(arrowpos, dd->sensordata /*  + 14 */);
    // Line end: w of right foot
    mju_copy3(diff, rf_gyr);
    // line length
    arrowsize[2] = mju_norm3(diff);

    // set mat to minimal rotation aligning b-a with z axis
    mju_quatZ2Vec(myquat, diff);
    mju_quat2Mat(myrot3x3_2, myquat);

    // one more geom to render
    scene->ngeom = scene->ngeom + 1;
    // mygeom now points to the last location in geoms buffer
    omegaR = scene->geoms + scene->ngeom - 1;
    // decor geom
    omegaR->objtype = mjOBJ_UNKNOWN;
    omegaR->objid = -1;
    omegaR->category = mjCAT_DECOR;
    omegaR->segid = scene->ngeom;
    // Add it to the scene
    mjv_initGeom(omegaR, mjGEOM_ARROW, arrowsize, arrowpos, myrot3x3_2, arrowrgba);

    //--------Plot the torso site Omega arrow----------------------------
    mjvGeom *omegaT;
    arrowrgba[0] = 0.9; // Red
    arrowrgba[1] = 0.;
    arrowrgba[2] = 0.;
    // Line start: at site torso
    mju_copy3(arrowpos, dd->sensordata);
    // Line end: w of torso
    mju_copy3(diff, torso_gyr);
    // line length
    arrowsize[2] = mju_norm3(diff);

    // set mat to minimal rotation aligning b-a with z axis
    mju_quatZ2Vec(myquat, diff);
    mju_quat2Mat(myrot3x3_2, myquat);

    // one more geom to render
    scene->ngeom = scene->ngeom + 1;
    // mygeom now points to the last location in geoms buffer
    omegaT = scene->geoms + scene->ngeom - 1;
    // decor geom
    omegaT->objtype = mjOBJ_UNKNOWN;
    omegaT->objid = -1;
    omegaT->category = mjCAT_DECOR;
    omegaT->segid = scene->ngeom;
    // Add it to the scene
    mjv_initGeom(omegaT, mjGEOM_ARROW, arrowsize, arrowpos, myrot3x3_2, arrowrgba);

    //--------Plot the tilting edge arrow----------------------------
    mjvGeom *tilt_edge;
    arrowrgba[0] = 0.;
    arrowrgba[1] = 0.9; // Green
    arrowrgba[2] = 0.;

    // Line start: at site torso (show at torso)
    mju_copy3(arrowpos, dd->sensordata);
    // Line end: tilt_direction (global coord)
    mju_copy3(diff, tilt_axis);
    // line length
    arrowsize[2] = mju_norm3(diff);

    // set mat to minimal rotation aligning b-a with z axis
    mju_quatZ2Vec(myquat, diff);
    mju_quat2Mat(myrot3x3_2, myquat);

    // one more geom to render
    scene->ngeom = scene->ngeom + 1;
    // mygeom now points to the last location in geoms buffer
    tilt_edge = scene->geoms + scene->ngeom - 1;
    // decor geom
    tilt_edge->objtype = mjOBJ_UNKNOWN;
    tilt_edge->objid = -1;
    tilt_edge->category = mjCAT_DECOR;
    tilt_edge->segid = scene->ngeom;
    // Add it to the scene
    mjv_initGeom(tilt_edge, mjGEOM_ARROW, arrowsize, arrowpos, myrot3x3_2, arrowrgba);

    mjv_addGeoms(mm, dd, opt, NULL, mjCAT_DECOR, scene);
}

void drawContactConvexHull(const mjModel *mm, mjData *dd, ContactDataAnalysis *cda, float rgba[], mjvScene *scene, const mjvOption *opt)
{
    double contact_convex_hull2D[10 * 2], convex_hull_polygon[10 * 3];
    int nvp1 = 0;
    double spa = 0;
    for (int i = 0; i < 10; i++)
        convex_hull_polygon[3 * i + 2] = 0; // Set z coord zero

    contact_cHull(mm, dd, contact_convex_hull2D, &nvp1, &spa);
    double height = 0 * cda->gcop[2];

    // printf("(in drawContactConvexHull) Skewness: %lg\n", skewness);
    for (int i = 0; i < nvp1; i++)
    {
        convex_hull_polygon[3 * i] = contact_convex_hull2D[2 * i];         // X coord copy
        convex_hull_polygon[3 * i + 1] = contact_convex_hull2D[2 * i + 1]; // Y coord copy
        convex_hull_polygon[3 * i + 2] = (height);                         // Set z coord equal to gen cop
        // abs(gcop[2]) becuase cannot see polygon if it goes below the ground
    }
    // printf("print n vertices: %d\n", nvp1);

    drawPolygon(mm, dd, rgba, scene, opt, convex_hull_polygon, nvp1 - 1);
}

void drawScaledContactConvexHull(const mjModel *mm, mjData *dd, ContactDataAnalysis *cda, float rgba[], mjvScene *scene, const mjvOption *opt)
{
    // Projectoin centred at COM.
    // The hull is drawn on the ground plane.

    // // Compute convex hull of the support polygon (assuming as if all the contact points were on the same plane).
    // double contact_convex_hull2D[15 * 2];
    // int nvp1 = 0; // Number of vertices of the convex hull + 1;
    // double spa = 0;
    // contact_cHull(mm, dd, contact_convex_hull2D, &nvp1, &spa);
    // double output_poygon[15 * 3];

    // List the foot contact points (with the ground).
    double foot_contact_points[15 * 3], projected_contact_points[15 * 3];
    for (int i = 0; i < cda->nconw; i++)
        mju_copy3(foot_contact_points + 3 * i, dd->contact[cda->world_con_IDs[i]].pos);

    // Project  them on the ground plane with the COM at focus.
    proj_contact_points(dd->subtree_com, foot_contact_points, cda->nconw, 0, projected_contact_points);

    // Find the convex hull of the projected contact points.
    double projected_contact_points_2D[15 * 2];
    for (int i = 0; i < cda->nconw; i++)
    {
        projected_contact_points_2D[2 * i] = projected_contact_points[3 * i];
        projected_contact_points_2D[2 * i + 1] = projected_contact_points[3 * i + 1];
    }
    double output_polygon_2D[15 * 3], spa;
    int nvp1; // Number of vertices of the convex hull + 1;
    cHull2D(projected_contact_points_2D, cda->nconw, output_polygon_2D, &nvp1, &spa);

    // Define the output polygon in 3D. Set z coord to zero (ground plane at the moment).
    double output_polygon[15 * 3];
    for (int i = 0; i < nvp1; i++)
    {
        output_polygon[3 * i] = output_polygon_2D[2 * i];
        output_polygon[3 * i + 1] = output_polygon_2D[2 * i + 1];
        output_polygon[3 * i + 2] = 0;
    }

    drawPolygon(mm, dd, rgba, scene, opt, output_polygon, nvp1 - 1);
}
/*--------MuJoCo Plotting-- mjvFigure---------*/

void plot_d3t_init(mjvFigure *myfig, char *mytitle, char *legend1, char *legend2, char *legend3, double last_nsec, double yrange[])
{
    // This function (initiates) plots three data streams w.r.t time like a ticker tape.

    // set figure to default
    mjv_defaultFigure(myfig);

    // title
    mju_strncpy(myfig->title, mytitle, 100);
    // x-label
    mju_strncpy(myfig->xlabel, "Time (s)", 10);

    // x-y tick number formats
    mju_strncpy(myfig->xformat, "%.3f", 20);
    mju_strncpy(myfig->yformat, "%.3f", 20);

    // legends
    mju_strncpy(myfig->linename[0], legend1, 50);
    mju_strncpy(myfig->linename[1], legend2, 50);
    mju_strncpy(myfig->linename[2], legend3, 50);

    // grid sizes
    myfig->gridsize[0] = 5;
    myfig->gridsize[1] = 5;

    // number of points on a line (works good only if all lines have same number of points)    myfig->linepnt[0] = 960 /* mjMAXLINEPNT */;
    myfig->linepnt[0] = 960 /* mjMAXLINEPNT */;
    myfig->linepnt[1] = 960 /* mjMAXLINEPNT */;
    myfig->linepnt[2] = 960 /* mjMAXLINEPNT */;

    // Set plot X-Y range
    myfig->range[0][0] = (float)-last_nsec; // Show time history for 'last_nsec' seconds
    myfig->range[0][1] = (float)0;          // Current time
    myfig->range[1][0] = (float)yrange[0];
    myfig->range[1][1] = (float)yrange[1];

    // Plot points (1000 max) on each line
    // We will have one line in this figure
    // The values will be 0 when the plot window appears
    for (int i = 0; i < myfig->linepnt[0]; i++)
    {
        // First line points: X0 Y0 X1 Y1 ... X99 Y99 ...X999 Y999
        myfig->linedata[0][2 * i] = -last_nsec + i * (last_nsec / (myfig->linepnt[0] - 1));
        myfig->linedata[0][2 * i + 1] = 0.;
        // Second line points: X0 Y0 X1 Y1 ... X99 Y99 ...X999 Y999
        myfig->linedata[1][2 * i] = -last_nsec + i * (last_nsec / (myfig->linepnt[0] - 1));
        myfig->linedata[1][2 * i + 1] = 0.;
        // Third line points: X0 Y0 X1 Y1 ... X99 Y99 ...X999 Y999
        myfig->linedata[2][2 * i] = -last_nsec + i * (last_nsec / (myfig->linepnt[0] - 1));
        myfig->linedata[2][2 * i + 1] = 0.;
    }
}

void plot_d2t_init(mjvFigure *myfig, char *mytitle, char *legend1, char *legend2, double last_nsec, double yrange[])
{
    // This function (initiates) plots two data streams w.r.t time like a ticker tape.

    // set figure to default
    mjv_defaultFigure(myfig);

    // title
    mju_strncpy(myfig->title, mytitle, 100);
    // x-label
    mju_strncpy(myfig->xlabel, "Time (s)", 10);

    // x-y tick number formats
    mju_strncpy(myfig->xformat, "%.3f", 20);
    mju_strncpy(myfig->yformat, "%.3f", 20);

    // legends
    mju_strncpy(myfig->linename[0], legend1, 50);
    mju_strncpy(myfig->linename[1], legend2, 50);

    // grid sizes
    myfig->gridsize[0] = 5;
    myfig->gridsize[1] = 5;

    // number of points on a line (works good only if two lines have same number of points)
    myfig->linepnt[0] = 960 /* mjMAXLINEPNT */;
    myfig->linepnt[1] = 960 /* mjMAXLINEPNT */;

    // Set plot X-Y range
    myfig->range[0][0] = (float)-last_nsec; // Show time history for 'last_nsec' seconds
    myfig->range[0][1] = (float)0;          // Current time
    myfig->range[1][0] = (float)yrange[0];
    myfig->range[1][1] = (float)yrange[1];

    // Plot points (1000 max) on each line
    // We will have one line in this figure
    // The values will be 0 when the plot window appears
    for (int i = 0; i < myfig->linepnt[0]; i++)
    {
        // First line points: X0 Y0 X1 Y1 ... X99 Y99 ...X999 Y999
        myfig->linedata[0][2 * i] = -last_nsec + i * (last_nsec / (myfig->linepnt[0] - 1));
        myfig->linedata[0][2 * i + 1] = 0.;
        // Second line points: X0 Y0 X1 Y1 ... X99 Y99 ...X999 Y999
        myfig->linedata[1][2 * i] = -last_nsec + i * (last_nsec / (myfig->linepnt[0] - 1));
        myfig->linedata[1][2 * i + 1] = 0.;
    }
}

void plot_d1t_init(mjvFigure *myfig, char *mytitle, char *legend1, double last_nsec, double yrange[])
{
    // This function (initiates) plots one data stream w.r.t time like a ticker tape.

    // set figure to default
    mjv_defaultFigure(myfig);

    // title
    mju_strncpy(myfig->title, mytitle, 100);
    // x-label
    mju_strncpy(myfig->xlabel, "Time (s)", 10);

    // x-y tick number formats
    mju_strncpy(myfig->xformat, "%.3f", 20);
    mju_strncpy(myfig->yformat, "%.3f", 20);

    // legends
    mju_strncpy(myfig->linename[0], legend1, 50);

    // grid sizes
    myfig->gridsize[0] = 5;
    myfig->gridsize[1] = 5;

    // number of points on a line
    myfig->linepnt[0] = 960 /* mjMAXLINEPNT */;

    // Set plot X-Y range
    myfig->range[0][0] = (float)-last_nsec; // Show time history for 'last_nsec' seconds
    myfig->range[0][1] = (float)0;          // Current time
    myfig->range[1][0] = (float)yrange[0];
    myfig->range[1][1] = (float)yrange[1];

    // Plot points (1000 max) on each line
    // We will have one line in this figure
    // The values will be 0 when the plot window appears
    for (int i = 0; i < myfig->linepnt[0]; i++)
    {
        // First line points: X0 Y0 X1 Y1 ... X99 Y99 ...X999 Y999
        myfig->linedata[0][2 * i] = -last_nsec + i * (last_nsec / (myfig->linepnt[0] - 1));
        myfig->linedata[0][2 * i + 1] = 0.;
    }
}

// template <typename T>
// void my_linspace_2d(T p1xy[], T p2xy[], int npoints, T opxy[])
// {
//     T dx = (p2xy[0] - p1xy[0]) / (npoints - 1);
//     T dy = (p2xy[1] - p1xy[1]) / (npoints - 1);

//     for (int i = 0; i < npoints; i++)
//     {
//         opxy[i * 2] = p1xy[0] + dx * i;     // x-coordinate
//         opxy[i * 2 + 1] = p1xy[1] + dy * i; // y-coordinate
//     }
// }

void plot_COP_init(mjvFigure *myfig)
{
    // This function (initiates) plots the projection of the generalised COP-ZOP and the foot support polygon.

    // set figure to default
    mjv_defaultFigure(myfig);

    // title
    char mytitle[1000] = "COP proj. trace";
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
        myfig->linepnt[i] = 100;
    }
    // the COP-ZMP projection trace
    myfig->linepnt[8] = 100 /* mjMAXLINEPNT */;

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

void plot_COP_update(mjvFigure *myfig, mjModel *mm, mjData *dd, ContactDataAnalysis *cda)
{
    // Some line joining bug when less than 8 contact points.
    // This function updates the projection of the generalised COP-ZOP and the foot support polygon.

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
                // Line joining the last point in contact group to the first point in the contact group
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
                    my_linspace_2d(FootVertices[cgrpId[i] + j], FootVertices[cgrpId[i] + j + 1], 500, myfig->linedata[k]);
                    k++;
                }
                // Line joining the last point in contact group to the first point in the contact group
                my_linspace_2d(FootVertices[cgrpId[i] + nc_ingrp[i] - 1], FootVertices[cgrpId[i]], 500, myfig->linedata[k]);
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
        int Np = 100;
        myfig->linepnt[k] = Np;

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
        for (int i = Np - 1; i > 1; i--)
        {
            myfig->linedata[k][2 * i] = myfig->linedata[k][2 * (i - 1)];
            myfig->linedata[k][2 * i + 1] = myfig->linedata[k][2 * (i - 1) + 1];
        }

        // Plot the latest COP-ZMP projection
        double gcz[3];
        mju_copy3(gcz, cda->gcop);
        myfig->linedata[k][2] = gcz[0];
        myfig->linedata[k][3] = gcz[1];
    }
}

void plot_d1t_update(mjvFigure *myfig, double last_nsec, double yrange[], double data1)
{
    // Currenty the function works good for 16 sec time history window

    // Reset y range if possible
    myfig->range[1][0] = (float)yrange[0];
    myfig->range[1][1] = (float)yrange[1];

    // Update the plotting data of one data stream
    // Last in first out
    // Refresh rate of 60 fps, show history of 'last_nsec' only
    int shift = (int)(myfig->linepnt[0] / (60 * last_nsec)); // useless at the moment

    for (int i = 0; i < (myfig->linepnt[0] - 1); i++)
    {
        // Line shift of y coordinates: 1 3 5 7 ... 998 (till second last point).

        myfig->linedata[0][2 * i + 1] = myfig->linedata[0][2 * (i + 1) + 1];
    }
    // Update the last point
    myfig->linedata[0][2 * (myfig->linepnt[0] - 1) + 1] = (float)data1;
}

void plot_d2t_update(mjvFigure *myfig, double last_nsec, double yrange[], double data1, double data2)
{
    // Currenty the function works good for 16 sec time history window
    // Works good if all lines have same number of points (e.g., 960)

    // Reset y range if possible
    myfig->range[1][0] = (float)yrange[0];
    myfig->range[1][1] = (float)yrange[1];

    // Update the plotting data of one data stream
    // Last in first out
    // Refresh rate of 60 fps, show history of 'last_nsec' only
    int shift = (int)(myfig->linepnt[0] / (60 * last_nsec)); // useless at the moment

    for (int i = 0; i < (myfig->linepnt[0] - 1); i++)
    {
        // Line shift of y coordinates: 1 3 5 7 ... 998 (till second last point).

        myfig->linedata[0][2 * i + 1] = myfig->linedata[0][2 * (i + 1) + 1];
        myfig->linedata[1][2 * i + 1] = myfig->linedata[1][2 * (i + 1) + 1];
    }
    // Update the last point
    myfig->linedata[0][2 * (myfig->linepnt[0] - 1) + 1] = (float)data1;
    myfig->linedata[1][2 * (myfig->linepnt[1] - 1) + 1] = (float)data2;
}

void plot_d3t_update(mjvFigure *myfig, double last_nsec, double yrange[], double data1, double data2, double data3)
{
    // Currenty the function works good for 16 sec time history window
    // Works good if all lines have same number of points (e.g., 960)

    // Reset y range if possible
    myfig->range[1][0] = (float)yrange[0];
    myfig->range[1][1] = (float)yrange[1];

    // Update the plotting data of one data stream
    // Last in first out
    // Refresh rate of 60 fps, show history of 'last_nsec' only
    int shift = (int)(myfig->linepnt[0] / (60 * last_nsec)); // useless at the moment

    for (int i = 0; i < (myfig->linepnt[0] - 1); i++)
    {
        // Line shift of y coordinates: 1 3 5 7 ... 998 (till second last point).

        myfig->linedata[0][2 * i + 1] = myfig->linedata[0][2 * (i + 1) + 1];
        myfig->linedata[1][2 * i + 1] = myfig->linedata[1][2 * (i + 1) + 1];
        myfig->linedata[2][2 * i + 1] = myfig->linedata[2][2 * (i + 1) + 1];
    }
    // Update the last point
    myfig->linedata[0][2 * (myfig->linepnt[0] - 1) + 1] = (float)data1;
    myfig->linedata[1][2 * (myfig->linepnt[1] - 1) + 1] = (float)data2;
    myfig->linedata[2][2 * (myfig->linepnt[2] - 1) + 1] = (float)data3;
}
