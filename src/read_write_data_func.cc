#include <cstdio>
#include <iostream>

#include "../include/mujoco/mujoco.h"
#include "my_mjc_func_declarations.h"
#include "rs485_bus_functions.h"
#include "exp_meas_proc_func.h"

void save_qpos(const mjModel *mm, mjData *dd)
{
    FILE *fid;
    fid = fopen("qpos_export.txt", "w");
    for (int i = 0; i < mm->nq; i++)
    {
        fprintf(fid, "%lg\n", dd->qpos[i]);
    }

    fclose(fid);
}

void load_qpos(const mjModel *mm, mjData *dd)
{
    FILE *fid;
    fid = fopen("qpos_export.txt", "r");
    int ferr = 0;
    if (fid != NULL)
    {
        for (int i = 0; i < mm->nq; i++)
        {
            ferr = fscanf(fid, "%lg\n", &(dd->qpos[i]));
            (!ferr) && (printf("error loading qpos from qpos_export.txt\n"));
        }

        fclose(fid);
    }
    else
    {
        printf("Error opening qpos_export.txt for reading\n");
    }
}

void save_sitepose(double ip_txyz[3], double ip_tkphi[4], double ip_lfxyz[3], double ip_lfkphi[4], double ip_rfxyz[3], double ip_rfkphi[4])
{
    FILE *fid;
    fid = fopen("site_export.txt", "w");


    // Torso site X Y Z
    for (int i = 0; i < 3; i++)
        fprintf(fid, "%lg\n", ip_txyz[i]);
    // Torso k phi
    for (int i = 0; i < 4; i++)
        fprintf(fid, "%lg\n", ip_tkphi[i]);

    // Left foot site X Y Z
    for (int i = 0; i < 3; i++)
        fprintf(fid, "%lg\n", ip_lfxyz[i]);
    // Left foot k phi
    for (int i = 0; i < 4; i++)
        fprintf(fid, "%lg\n", ip_lfkphi[i]);

    // Right foot site X Y Z
    for (int i = 0; i < 3; i++)
        fprintf(fid, "%lg\n", ip_rfxyz[i]);
    // Right foot k phi
    for (int i = 0; i < 4; i++)
        fprintf(fid, "%lg\n", ip_rfkphi[i]);

    fclose(fid);
}

void save_com_pos(double ip_com_pos[3])
{
    FILE *fid;
    fid = fopen("com_export.txt", "w");

    // Torso site X Y Z
    for (int i = 0; i < 3; i++)
        fprintf(fid, "%lg\n", ip_com_pos[i]);
}

void load_sitepose(double op_txyz[3], double op_tkphi[4], double op_lfxyz[3], double op_lfkphi[4], double op_rfxyz[3], double op_rfkphi[4])
{
    extern double txyz[3], tkphi[4], lfxyz[3], lfkphi[4], rfxyz[3], rfkphi[4];
    FILE *fid;
    fid = fopen("site_export.txt", "r");
    int ferr = 0;

    if (fid != NULL)
    {
        // Torso site X Y Z
        for (int i = 0; i < 3; i++)
            ferr = fscanf(fid, "%lg\n", &txyz[i]);
        // Torso k phi
        for (int i = 0; i < 4; i++)
            ferr = fscanf(fid, "%lg\n", &tkphi[i]);

        // Left foot site X Y Z
        for (int i = 0; i < 3; i++)
            ferr = fscanf(fid, "%lg\n", &lfxyz[i]);
        // Left foot k phi
        for (int i = 0; i < 4; i++)
            ferr = fscanf(fid, "%lg\n", &lfkphi[i]);

        // Right foot site X Y Z
        for (int i = 0; i < 3; i++)
            ferr = fscanf(fid, "%lg\n", &rfxyz[i]);
        // Right foot k phi
        for (int i = 0; i < 4; i++)
            ferr = fscanf(fid, "%lg\n", &rfkphi[i]);

        fclose(fid);
    }
    else
    {
        printf("Error opening site_export.txt for reading\n");
    }
}

void load_com_pos(double op_com_pos[3])
{
    extern double com_pos[3];
    FILE *fid;
    fid = fopen("com_export.txt", "r");
    int ferr = 0;

    if (fid != NULL)
    {
        // Centre of mass X Y Z
        for (int i = 0; i < 3; i++)
            ferr = fscanf(fid, "%lg\n", &op_com_pos[i]);

        fclose(fid);
    }
}

void save_contact_points(const mjModel *mm, mjData *dd)
{
    FILE *fid;
    fid = fopen("contact_points.txt", "w");
    for (int i = 0; i < dd->ncon; i++)
    {
        fprintf(fid, "%lg\t%lg\t%lg\n", dd->contact[i].pos[0], dd->contact[i].pos[1], dd->contact[i].pos[2]);
    }

    fclose(fid);
}

void append_data_vector_to_file(FILE *fp, double data[], int len)
{
    // Append a row of incoming vector data to this file

    for (int i = 0; i < len; i++)
    {
        fprintf(fp, "%lg ", data[i]);
    }
    fprintf(fp, "\n");

}

void append_time_data_vector_to_file(FILE *fp, double time, double data[], int len)
{
    // Append a row of incoming vector data to this file

    fprintf(fp, "%lg ", time);
    for (int i = 0; i < len; i++)
    {
        fprintf(fp, "%lg ", data[i]);
    }
    fprintf(fp, "\n");
}

void record_vec12_data(double data[], FILE *fp)
{
    // Write simulation data to file:
    append_data_vector_to_file(fp, data, 12); // Write to file.
}

void record_tvec3_data(const mjModel *mm, mjData *dd, double data[], FILE *fp)
{
    // Write simulation data to file:
    append_time_data_vector_to_file(fp, dd->time, data, 3); // Write to file.
}
