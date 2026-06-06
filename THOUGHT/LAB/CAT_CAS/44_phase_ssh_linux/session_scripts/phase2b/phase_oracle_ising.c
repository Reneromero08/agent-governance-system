#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>

#define S 30
uint64_t lcg(uint64_t *s){*s=(*s*0x41C64E6D+0x3039);*s=(*s>>13)^*s;*s=(*s<<17)+*s;return *s;}
void decode(double *t,uint64_t *s,int N){for(int i=0;i<N;i++)s[i]=(cos(t[i])>=0)?1:0;}
int64_t ising_n(uint64_t *s,int64_t *J,int N){int64_t E=0;for(int i=0;i<N;i++){int si=(s[i]&1)?1:-1;for(int j=i+1;j<N;j++){int sj=(s[j]&1)?1:-1;E-=J[i*N+j]*si*sj;}}return E;}
void v7_descend_n(double *t,int64_t *J,int N,uint64_t *rng){
    for(int st=0;st<S;st++){int i=lcg(rng)%N;double g=0;for(int j=0;j<N;j++)if(J[i*N+j])g+=J[i*N+j]*sin(t[i]-t[j]);t[i]-=0.1*g;}
}
void v11_descend_n(double *t,int64_t *J,int *edg,int ne,int N,uint64_t *rng){
    for(int st=0;st<S;st++){int i=lcg(rng)%N;double g=0;for(int j=0;j<N;j++)if(J[i*N+j])g+=J[i*N+j]*sin(t[i]-t[j]);
        double cg=0;for(int e=0;e<ne;e++){int a=edg[e*3],b=edg[e*3+1],sg=edg[e*3+2];if(a==i)cg+=sg*sin(t[i]-t[b]);if(b==i)cg+=sg*sin(t[i]-t[a]);}t[i]-=0.1*(g+0.5*cg/ne);}
    for(int pass=0;pass<5;pass++){int worst=-1;double wa=1e9;for(int e=0;e<ne;e++){int a=edg[e*3],b=edg[e*3+1],sg=edg[e*3+2];double ag=sg*cos(t[a]-t[b]);if(ag<wa){wa=ag;worst=e;}}if(wa>0.3)break;int a=edg[worst*3],b=edg[worst*3+1],sg=edg[worst*3+2];t[a]+=0.3*(((sg>0)?0:M_PI)-(t[a]-t[b]));}
}
void gen_sparse(int64_t *J,int N,int nedges,uint64_t *rs){
    memset(J,0,N*N*8);for(int e=0;e<nedges;e++){int a=lcg(rs)%N,b=lcg(rs)%N;if(a!=b&&!J[a*N+b]){int v=(lcg(rs)&1)?1:-1;J[a*N+b]=J[b*N+a]=v;}}
}
void gen_frust(int64_t *J,int N,uint64_t *rs){
    memset(J,0,N*N*8);int nodes[3];for(int k=0;k<3;k++)nodes[k]=lcg(rs)%N;
    J[nodes[0]*N+nodes[1]]=J[nodes[1]*N+nodes[0]]=1;
    J[nodes[1]*N+nodes[2]]=J[nodes[2]*N+nodes[1]]=1;
    J[nodes[0]*N+nodes[2]]=J[nodes[2]*N+nodes[0]]=1;
    for(int i=1;i<N-1;i++)J[i*N+(i+1)]=J[(i+1)*N+i]=1;
    for(int e=0;e<N/2;e++){int a=lcg(rs)%N,b=lcg(rs)%N;if(a!=b&&!J[a*N+b]){int v=(lcg(rs)&1)?1:-1;J[a*N+b]=J[b*N+a]=v;}}
}

typedef struct{double best,mean;int hits;}st;
void s64(int64_t *E,int n,int64_t g,st *o){o->hits=0;o->best=1e9;double s=0;for(int i=0;i<n;i++){if(E[i]<o->best)o->best=E[i];if(E[i]==g)o->hits++;s+=E[i];}o->mean=s/n;}

int main(){
    printf("=== PHASE 2B.5A FINAL KILL SHOT: N=24/N=32 STRESS TEST ===\n");
    printf("Energy-ensemble: v7+v11 same seed, pick lower energy.\n\n");

    uint64_t rs=0xCADE000000000001ULL;
    int sizes[]={24,32},paths[]={100,30};
    int n_sizes=2;
    for(int si=0;si<n_sizes;si++){
        int N=sizes[si],cp=paths[si];
        printf("========================================\n");
        printf("N=%d (%d paths)\n",N,cp);
        printf("========================================\n\n");

        for(int pt=0;pt<3;pt++){
            int64_t J[N*N];const char *nm;
            if(pt==0){gen_sparse(J,N,N+2,&rs);nm="RandSparse";}
            else if(pt==1){gen_frust(J,N,&rs);nm="Frustrated";}
            else{memset(J,0,N*N*8);int sp[N];for(int i=0;i<N;i++)sp[i]=i%2;
                 for(int i=0;i<N;i++)for(int j=i+1;j<N;j++)J[i*N+j]=J[j*N+i]=(sp[i]==sp[j])?1:-1;nm="Planted";}

            int edg[512][3],ne=0;for(int i=0;i<N;i++)for(int j=i+1;j<N;j++)if(J[i*N+j]){edg[ne][0]=i;edg[ne][1]=j;edg[ne][2]=J[i*N+j];ne++;}
            printf("=== %s N=%d (%d edges) ===\n",nm,N,ne);

            int64_t *Ev7=malloc(cp*8),*Ev11=malloc(cp*8),*Eens=malloc(cp*8),*Eew=malloc(cp*8),*Ersp=malloc(cp*8),*Erpd=malloc(cp*8),*Essh=malloc(cp*8);
            uint64_t rg=0x1111000000000001ULL+pt*0x1000;

            int64_t Jrw[N*N];memset(Jrw,0,N*N*8);
            {int ea[512],eb[512];for(int e=0;e<ne;e++){ea[e]=edg[e][0];eb[e]=edg[e][1];}
            for(int e=0;e<ne;e++){int j=lcg(&rg)%ne;int t=ea[e];ea[e]=ea[j];ea[j]=t;t=eb[e];eb[e]=eb[j];eb[j]=t;}
            for(int e=0;e<ne;e++){int v=J[edg[e][0]*N+edg[e][1]];Jrw[ea[e]*N+eb[e]]=Jrw[eb[e]*N+ea[e]]=v;}}
            int64_t Jsh[N*N];memcpy(Jsh,J,N*N*8);
            for(int e=0;e<ne;e++){int a=edg[e][0],b=edg[e][1];
                int v=lcg(&rg)&1?1:-1;Jsh[a*N+b]=Jsh[b*N+a]=v;}

            for(int f=0;f<cp;f++){
                double t0[N];for(int i=0;i<N;i++)t0[i]=2*M_PI*(lcg(&rg)/(double)UINT64_MAX);
                uint64_t sp[N];double tv[N],t11[N];memcpy(tv,t0,N*8);memcpy(t11,t0,N*8);
                v7_descend_n(tv,J,N,&rg);decode(tv,sp,N);Ev7[f]=ising_n(sp,J,N);
                v11_descend_n(t11,J,(int*)edg,ne,N,&rg);decode(t11,sp,N);Ev11[f]=ising_n(sp,J,N);
                Eens[f]=(Ev7[f]<Ev11[f])?Ev7[f]:Ev11[f];
                double trw[N];for(int i=0;i<N;i++)trw[i]=2*M_PI*(lcg(&rg)/(double)UINT64_MAX);
                v7_descend_n(trw,Jrw,N,&rg);decode(trw,sp,N);Eew[f]=ising_n(sp,J,N);
                for(int i=0;i<N;i++)sp[i]=lcg(&rg)&1;
                Ersp[f]=ising_n(sp,J,N);
                double trpd[N];for(int i=0;i<N;i++)trpd[i]=2*M_PI*(lcg(&rg)/(double)UINT64_MAX);
                v7_descend_n(trpd,J,N,&rg);decode(trpd,sp,N);Erpd[f]=ising_n(sp,J,N);
                double tsh[N];for(int i=0;i<N;i++)tsh[i]=2*M_PI*(lcg(&rg)/(double)UINT64_MAX);
                v7_descend_n(tsh,Jsh,N,&rg);decode(tsh,sp,N);Essh[f]=ising_n(sp,J,N);
            }

            st sv7,sv11,sens,sew,srsp,srpd,sssh;s64(Ev7,cp,INT64_MAX,&sv7);s64(Ev11,cp,INT64_MAX,&sv11);
            s64(Eens,cp,INT64_MAX,&sens);s64(Eew,cp,INT64_MAX,&sew);
            s64(Ersp,cp,INT64_MAX,&srsp);s64(Erpd,cp,INT64_MAX,&srpd);s64(Essh,cp,INT64_MAX,&sssh);
            printf("  v7:        best=%6.0f mean=%+8.2f\n",sv7.best,sv7.mean);
            printf("  v11:       best=%6.0f mean=%+8.2f\n",sv11.best,sv11.mean);
            printf("  ENSEMBLE:  best=%6.0f mean=%+8.2f\n",sens.best,sens.mean);
            printf("  Edge-rewired: best=%6.0f mean=%+8.2f\n",sew.best,sew.mean);
            printf("  RandSpin:  best=%6.0f mean=%+8.2f\n",srsp.best,srsp.mean);
            printf("  RandPhaseDesc: best=%6.0f mean=%+8.2f\n",srpd.best,srpd.mean);
            printf("  SignShuffled: best=%6.0f mean=%+8.2f\n",sssh.best,sssh.mean);
            printf("  Ensemble vs v7: %+.2f  vs v11: %+.2f  vs rewired: %+.2f  vs randsp: %+.2f  vs rpd: %+.2f\n\n",
                sens.mean-sv7.mean,sens.mean-sv11.mean,sens.mean-sew.mean,sens.mean-srsp.mean,sens.mean-srpd.mean);
            free(Ev7);free(Ev11);free(Eens);free(Eew);free(Ersp);free(Erpd);free(Essh);
        }
    }
    printf("=== VERDICT ===\n");
    printf("Final kill shot: N=%d (%d paths).\n",sizes[n_sizes-1],paths[n_sizes-1]);
    printf("Energy-ensemble vs v7, v11, edge-rewired, rand spin, rand phase descent.\n");
    printf("PHASE2B_5A_FINAL_KILLSHOT_IMPLEMENTED\n");
    return 0;
}
