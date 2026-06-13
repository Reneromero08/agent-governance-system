#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
#include <limits.h>
#include "catcas_phase3.h"

#define NN 8
#define PATHS 100
#define S 20

uint64_t lcg(uint64_t *s){*s=(*s*0x41C64E6D+0x3039);*s=(*s>>13)^*s;*s=(*s<<17)+*s;return *s;}
int64_t ising64(uint64_t *sp,int64_t (*J)[NN]){int64_t E=0;for(int i=0;i<NN;i++){int si=(sp[i]&1)?1:-1;for(int j=i+1;j<NN;j++){int sj=(sp[j]&1)?1:-1;E-=J[i][j]*si*sj;}}return E;}
void brute_ising(int64_t (*J)[NN],int64_t *g){int64_t b=INT64_MAX;for(uint64_t m=0;m<(1ULL<<NN);m++){uint64_t s[NN];for(int i=0;i<NN;i++)s[i]=(m>>i)&1;int64_t E=ising64(s,J);if(E<b)b=E;}*g=b;}

void vertex_descend(double *theta,int64_t (*J)[NN],uint64_t *rng){
    for(int st=0;st<S;st++){int i=lcg(rng)%NN;double g=0;
        for(int j=0;j<NN;j++)if(J[i][j])g+=J[i][j]*sin(theta[i]-theta[j]);
        theta[i]-=0.1*g;}
}
void decode_spins(double *t,uint64_t *s){for(int i=0;i<NN;i++)s[i]=(cos(t[i])>=0)?1:0;}

int solve_edge(uint64_t *sp,int *edges,int ne,int maxit){
    for(int it=0;it<maxit;it++){int imp=0;
        for(int e=0;e<ne;e++){int i=edges[e*3],j=edges[e*3+1],sg=edges[e*3+2];
            if(sg>0&&(sp[i]&1)!=(sp[j]&1)){sp[i]^=1;imp=1;}
            else if(sg<0&&(sp[i]&1)==(sp[j]&1)){sp[i]^=1;imp=1;}}
        if(!imp)break;}
    return 0;
}

typedef struct{double best,mean,std;int hits,restores;} st;
void stats64(int64_t *E,int n,int64_t g,st *o,int rst){o->hits=0;o->best=1e9;double s=0;
    for(int i=0;i<n;i++){if(E[i]<o->best)o->best=E[i];if(E[i]==g)o->hits++;s+=E[i];}
    o->mean=s/n;double v=0;for(int i=0;i<n;i++)v+=(E[i]-o->mean)*(E[i]-o->mean);o->std=sqrt(v/n);o->restores=rst;}

int main(){
    printf("=== PHASE 3.14: HYBRID PHASE-SEEDED CATALYTIC ISING ===\n\n");
    printf("Pipeline: phase oracle seed → decode → active refine → catcas restore\n");
    printf("Baselines: active-only, phase-only, random-seeded active\n\n");

    const char *nm[]={"Ferro","Anti-ferro","Mixed","RandSparse","FrustCycle","Planted"};
    int64_t Js[6][NN][NN]={{{0}}};int64_t gs[6];int edges[6][32][3],nec[6];
    for(int i=0;i<NN-1;i++){Js[0][i][i+1]=Js[0][i+1][i]=1;}
    for(int i=0;i<NN-1;i++){Js[1][i][i+1]=Js[1][i+1][i]=-1;}
    for(int i=0;i<NN-1;i++){Js[2][i][i+1]=Js[2][i+1][i]=(i%2)?-1:1;}
    uint64_t rs=0xDEAD000000000001ULL;
    for(int e=0;e<10;e++){int a=lcg(&rs)%NN,b=lcg(&rs)%NN;if(a!=b)Js[3][a][b]=Js[3][b][a]=(lcg(&rs)&1)?1:-1;}
    Js[4][0][1]=Js[4][1][0]=1;Js[4][1][2]=Js[4][2][1]=1;Js[4][0][2]=Js[4][2][0]=1;
    for(int i=2;i<NN-1;i++)Js[4][i][i+1]=Js[4][i+1][i]=1;
    int ss[NN]={1,0,1,0,1,0,1,0};for(int i=0;i<NN;i++)for(int j=i+1;j<NN;j++)Js[5][i][j]=Js[5][j][i]=(ss[i]==ss[j])?1:-1;
    for(int p=0;p<6;p++){brute_ising(Js[p],&gs[p]);nec[p]=0;for(int i=0;i<NN;i++)for(int j=i+1;j<NN;j++)if(Js[p][i][j]){edges[p][nec[p]][0]=i;edges[p][nec[p]][1]=j;edges[p][nec[p]][2]=Js[p][i][j];nec[p]++;}}

    for(int p=0;p<6;p++){
        printf("=== %s (%d edges, ground=%ld) ===\n",nm[p],nec[p],(long)gs[p]);
        uint64_t rng=0x1111000000000001ULL+p*0x1000;
        int64_t E_phase[PATHS],E_act[PATHS],E_hybrid[PATHS],E_rndact[PATHS],E_pre[PATHS];
        int rest_ok=0;

        for(int path=0;path<PATHS;path++){
            // Phase seed generation
            double theta[NN];for(int i=0;i<NN;i++)theta[i]=2*M_PI*(lcg(&rng)/(double)UINT64_MAX);
            vertex_descend(theta,Js[p],&rng);
            uint64_t seed[NN];decode_spins(theta,seed);
            E_phase[path]=ising64(seed,Js[p]);

            // Hybrid: refine phase seed with active solver
            catcas_tape_t *t=catcas_tape_init();
            unsigned char h0[CATCAS_SHA256_LEN];
            for(int i=0;i<NN;i++)catcas_slot_write(t,i,seed[i]);
            catcas_tape_snapshot(t,h0);
            uint64_t sp[NN];for(int i=0;i<NN;i++)sp[i]=catcas_slot_read(t,i)&1;
            solve_edge(sp,(int*)edges[p],nec[p],5000);
            int64_t Ehy=ising64(sp,Js[p]);E_hybrid[path]=Ehy;
            for(int i=0;i<NN;i++)catcas_slot_write(t,i,seed[i]);
            if(catcas_tape_verify(t,h0))rest_ok++;
            catcas_tape_destroy(t);

            // Pre-refine energy = phase seed energy
            E_pre[path]=E_phase[path];

            // Active baseline: random init + solve
            for(int i=0;i<NN;i++)sp[i]=lcg(&rng)&1;
            solve_edge(sp,(int*)edges[p],nec[p],5000);
            E_act[path]=ising64(sp,Js[p]);

            // Random-seeded active: random init (different seed) + solve  
            for(int i=0;i<NN;i++)sp[i]=lcg(&rng)&1;
            solve_edge(sp,(int*)edges[p],nec[p],5000);
            E_rndact[path]=ising64(sp,Js[p]);
        }

        st sph,sact,shyb,srnd,spre;
        stats64(E_phase,PATHS,gs[p],&sph,0);
        stats64(E_act,PATHS,gs[p],&sact,0);
        stats64(E_hybrid,PATHS,gs[p],&shyb,rest_ok);
        stats64(E_rndact,PATHS,gs[p],&srnd,0);
        stats64(E_pre,PATHS,gs[p],&spre,0);

        printf("  %-22s best=%3.0f mean=%+6.2f hits=%3d/%d\n","Phase seed only",sph.best,sph.mean,sph.hits,PATHS);
        printf("  %-22s best=%3.0f mean=%+6.2f hits=%3d/%d\n","Active only",sact.best,sact.mean,sact.hits,PATHS);
        printf("  %-22s best=%3.0f mean=%+6.2f hits=%3d/%d\n","Random+active",srnd.best,srnd.mean,srnd.hits,PATHS);
        printf("  %-22s best=%3.0f mean=%+6.2f hits=%3d/%d\n","HYBRID phase+active",shyb.best,shyb.mean,shyb.hits,PATHS);
        printf("  Hybrid pre-refine: mean=%+6.2f\n",spre.mean);
        printf("  Hybrid vs Active: mean=%+.2f hits=%+d\n",shyb.mean-sact.mean,shyb.hits-sact.hits);
        printf("  Hybrid vs Random+active: mean=%+.2f hits=%+d\n",shyb.mean-srnd.mean,shyb.hits-srnd.hits);
        printf("  Restore: %d/%d\n\n",shyb.restores,PATHS);
    }

    printf("=== VERDICT ===\n");
    printf("Hybrid: phase oracle seed → active edge refinement → catcas restore.\n");
    printf("If hybrid beats random+active: phase seeding improves basin selection.\n");
    printf("If hybrid = active: phase seed adds no value over random init.\n");
    printf("PHASE3_14_HYBRID_PHASE_SEEDED_ISING_COMPLETE\n");
    return 0;
}
