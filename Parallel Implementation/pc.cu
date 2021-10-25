#include <iostream>
#include <cstdlib>
#include <math.h>
#include <omp.h>

#include "kMeans.cu"

using namespace std;

int findSide(int p1x, int p1y, int p2x, int p2y, int px, int py)
{
    int val = (py - p1y) * (p2x - p1x) -
        (p2y - p1y) * (px - p1x);

    if (val > 0)
        return 1;
    if (val < 0)
        return -1;
    return 0;
}

// returns a value proportional to the distance
// between the point p and the line joining the
// points p1 and p2
int lineDist(int px, int py, int qx, int qy, int rx, int ry)
{
    return abs((ry - py) * (qx - px) - (qy - py) * (rx - px));
}

int setfunction(int hullx[], int hully[], int index, int convx[], int convy[]) {
    int ind = 0, temp;
    for (int i = 0; i < index - 1; i++) {
        for (int j = 0; j < index - i - 1; j++) {
            if (hullx[j] > hullx[j + 1]) {
                temp = hullx[j];
                hullx[j] = hullx[j + 1];
                hullx[j + 1] = temp;

                temp = hully[j];
                hully[j] = hully[j + 1];
                hully[j + 1] = temp;
            }
            else if (hullx[j] == hullx[j + 1]) {
                if (hully[j] > hully[j + 1]) {
                    temp = hully[j];
                    hully[j] = hully[j + 1];
                    hully[j + 1] = temp;
                }
            }
        }
    }
    int i = 0;
    while (i < index) {
        convx[ind] = hullx[i];
        convy[ind] = hully[i];
        ind++;
        if (hullx[i] == hullx[i + 1] && hully[i] == hully[i + 1])
            i = i + 2;
        else
            i++;
    }
    return ind;
}

// End points of line L are p1 and p2.  side can have value
// 1 or -1 specifying each of the parts made by the line L
int quickHull(int x[], int y[], int n, int px, int py, int qx, int qy, int side, int hullx[], int hully[], int index)
{
    int ind = -1;
    int max_dist = 0;

    // finding the point with maximum distance
    // from L and also on the specified side of L.
    for (int i = 0; i < n; i++)
    {
        int temp = lineDist(px, py, qx, qy, x[i], y[i]);
        if (findSide(px, py, qx, qy, x[i], y[i]) == side && temp > max_dist)
        {
            ind = i;
            max_dist = temp;
        }
    }

    // If no point is found, add the end points
    // of L to the convex hull.
    if (ind == -1)
    {
        hullx[index] = px;
        hully[index] = py;
        index = index + 1;
        hullx[index] = qx;
        hully[index] = qy;
        index = index + 1;
        return index;
    }

    // Recur for the two parts divided by a[ind]
    index = quickHull(x, y, n, x[ind], y[ind], px, py, - findSide(x[ind], y[ind], px, py, qx, qy), hullx, hully, index);
    index = quickHull(x, y, n, x[ind], y[ind], qx, qy, - findSide(x[ind], y[ind], qx, qy, px, py), hullx, hully, index);
    return index;
}

void findHull(int x[], int y[], int n, int* convx, int* convy, int* n_c)
{
    printf("\n\nCluster Size %d;\n", n);
    int* hullx = new int[10000000];
    int* hully = new int[10000000];


    // int hullx[10000000], hully[10000000];
    int index = 0;

    if (n < 3)
    {
        printf("Convex hull not possible\n");
        return;
    }


    int min_x = 0, max_x = 0;
    for (int i = 1; i < n; i++)
    {
        if (x[i] < x[min_x])
            min_x = i;
        if (x[i] > x[max_x])
            max_x = i;
    }

    index = quickHull(x, y, n, x[min_x], y[min_x], x[max_x], y[max_x], 1, hullx, hully, index);


    index = quickHull(x, y, n, x[min_x], y[min_x], x[max_x], y[max_x], -1, hullx, hully, index);

    
    int ind = setfunction(hullx, hully, index, convx, convy);
    *n_c = ind;
    

}



int main()
{
    //Read the 2d points from a file
    const int n = 5000, k = 3;
    double t1, t2, t3, t4;
    int* x = (int*)malloc(n * sizeof(int));
    int* y = (int*)malloc(n * sizeof(int));
    int* convx = (int*)malloc(n * sizeof(int));
    int* convy = (int*)malloc(n * sizeof(int));
    int* kc = (int*)malloc(n * sizeof(int));
    // int x[n], y[n], convx[n], convy[n], kc[n];
    srand(time(0));
    cout << "Generating Random Numbers...!!!" << endl;
    t1 = omp_get_wtime();
    for (int i = 0; i < n; i++) {
        x[i] = rand() % n;
        y[i] = rand() % n;
    }

    //call the clusters' function and get the clusters
    cout << "Obtaining the clusters...!!" << endl;

    t3 = omp_get_wtime();
    
    find_clusters(n, x, y, kc);

    t4 = omp_get_wtime();

    printf("\nTime Taken for K-Means Cluster :%lf \n", t4 - t3);

    /*for (int i = 0; i < n; i++) {
        cout << kc[i] << ' ';
    }*/
    auto xc = new int[k][n];
    auto yc = new int[k][n];
    int clust_size[k];

    // int yc[k][n], xc[k][n], clust_size[k];

    for (int p = 0; p < k; p++) {
        clust_size[p] = 0;
    }
 
    for (int j = 0; j < k; j++) {
        //printf("Cluster %d;\n", j);
        for (int i = 0; i < n; i++) {
            if (kc[i] == j) {
                //printf("point %d: (%d,%d)\n", i, x[i], y[i]);
                clust_size[j] += 1;
                xc[j][clust_size[j] - 1] = x[i];
                yc[j][clust_size[j] - 1] = y[i];
                //cout<<x[i]<<","<<y[i]<<endl;
            }
        }
    }

    for (int i = 0; i < k; i++) {
        printf("clust_size : %d ", clust_size[i]);
    }

    cout << "Finding the convex hulls for each cluster obtained..!!" << endl;
    //for each cluster find the convex hull and store the points

    /*int* con_x = new int[n];
    int* con_y = new int[n];
    int* fin_x = new int[n];
    int* fin_y = new int[n];*/

    int con_x[n], con_y[n], con_n = 0, fin_x[n], fin_y[n], fin_n = 0;
    

    for (int j = 0; j < k; j++) {
        printf("\n\nCluster %d;\n", j);
        findHull(xc[j], yc[j], clust_size[j], con_x, con_y, &con_n);
        for (int i = 0; i < con_n; i++) {
            fin_x[fin_n + i] = con_x[i];
            fin_y[fin_n + i] = con_y[i];
            printf("(%d,%d)\n", con_x[i], con_y[i]);
        }
        fin_n += con_n;
        printf("l_n=%d,g_n=%d\n", con_n, fin_n);
    }

    //now find the convex hull for the points stored above
    printf("\n\nFinding the final Hull..!!\n");

    for (int i = 0; i < fin_n; i++) {
        printf("(%d,%d)\n", fin_x[i], fin_y[i]);
    }
    printf("%d\n", fin_n);

    findHull(fin_x, fin_y, fin_n, con_x, con_y, &con_n);

    printf("***\n");

    for (int i = 0; i < con_n; i++) {
        printf("(%d,%d)\n", con_x[i], con_y[i]);
    }
    t2 = omp_get_wtime();
    printf("Time Taken:%lf\n", t2 - t1);
    return 0;
}