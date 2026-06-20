#define _GNU_SOURCE
#include "combined_pdn_hardware.h"

#include <ctype.h>
#include <errno.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/wait.h>
#include <unistd.h>

#define SHA_LEN 64

static void die(const char *fmt, ...) {
    va_list ap; va_start(ap, fmt); fputs("ERROR: ", stderr); vfprintf(stderr, fmt, ap);
    fputc('\n', stderr); va_end(ap); exit(2);
}
static int path_join(char *out,size_t n,const char *a,const char *b){
    if(!a||!b||b[0]=='/'||strstr(b,"..")) return -1;
    int k=snprintf(out,n,"%s/%s",a,b); return k<0||(size_t)k>=n?-1:0;
}
static long file_size(const char *p){struct stat s;return stat(p,&s)||!S_ISREG(s.st_mode)?-1:(long)s.st_size;}
static int exists(const char *p){struct stat s;return stat(p,&s)==0;}
static char *slurp(const char *p){
    FILE*f=fopen(p,"rb"); if(!f)die("open %s: %s",p,strerror(errno));
    if(fseek(f,0,SEEK_END)) die("seek %s",p);
    long z=ftell(f);
    if(z<0||fseek(f,0,SEEK_SET)) die("seek %s",p);
    char*b=calloc((size_t)z+1,1); if(!b)die("oom"); if(z&&fread(b,1,(size_t)z,f)!=(size_t)z)die("read %s",p); fclose(f); return b;
}
static const char *key(const char*j,const char*k){char n[160];snprintf(n,sizeof(n),"\"%s\"",k);return strstr(j,n);}
static const char *value(const char*j,const char*k){const char*p=key(j,k);if(!p||(p=strchr(p,':'))==NULL)return NULL;do{p++;}while(isspace((unsigned char)*p));return p;}
static int jstr(const char*j,const char*k,char*out,size_t n){const char*p=value(j,k);if(!p||*p!='\"')return-1;const char*q=strchr(++p,'\"');size_t z=q?(size_t)(q-p):n;if(!q||!z||z>=n)return-1;memcpy(out,p,z);out[z]=0;return 0;}
static int jlong(const char*j,const char*k,long*out){const char*p=value(j,k);if(!p)return-1;char*e;long v=strtol(p,&e,10);if(e==p)return-1;*out=v;return 0;}
static int jbool(const char*j,const char*k,int*out){const char*p=value(j,k);if(!p)return-1;if(!strncmp(p,"true",4)){*out=1;return 0;}if(!strncmp(p,"false",5)){*out=0;return 0;}return-1;}
static int jnullable_long(const char*j,const char*k,long*out,int*present){const char*p=value(j,k);if(!p)return-1;if(!strncmp(p,"null",4)){*present=0;*out=0;return 0;}*present=1;return jlong(j,k,out);}
static int shell_safe(const char*p){if(!p||!*p||strstr(p,".."))return 0;for(;*p;p++)if((unsigned char)*p<32||strchr("';&|`$<>",*p))return 0;return 1;}
static void sha256(const char*p,char out[65]){
    if(!shell_safe(p)) die("unsafe path: %s",p);
    char cmd[CP_PATH_MAX+32];
    if(snprintf(cmd,sizeof(cmd),"sha256sum '%s'",p)>=(int)sizeof(cmd)) die("path too long");
    FILE*f=popen(cmd,"r");char line[128];if(!f||!fgets(line,sizeof(line),f))die("sha256 failed");int rc=pclose(f);if(rc==-1||!WIFEXITED(rc)||WEXITSTATUS(rc))die("sha256 failed");
    for(int i=0;i<SHA_LEN;i++){if(!isxdigit((unsigned char)line[i]))die("invalid sha256 output");out[i]=(char)tolower((unsigned char)line[i]);}out[64]=0;
}
static void verify_file(const char*m,const char*root,const char*name){
    const char*e=key(m,name);long want;char wh[65],p[CP_PATH_MAX],got[65];if(!e)die("manifest missing %s",name);
    if(jlong(e,"size",&want)||jstr(e,"sha256",wh,sizeof(wh))||strlen(wh)!=64)die("invalid manifest entry %s",name);
    if(path_join(p,sizeof(p),root,name)||file_size(p)!=want) die("size mismatch for %s",name);
    sha256(p,got);
    if(strcmp(wh,got)) die("sha256 mismatch for %s",name);
}
static RunnerArgs parse_args(int ac,char**av){
    RunnerArgs a={0};a.victim=a.sender=-1;a.pin_khz=1600000;a.slot_s=a.off_window_s=.5;a.read_hz=4000;a.temp_veto_c=68;a.backend=BACKEND_REAL;
    for(int i=1;i<ac;i++){const char*k=av[i],*v=i+1<ac?av[i+1]:NULL;
        if(!strcmp(k,"--validate-only"))a.mode=MODE_VALIDATE;
        else if(!strcmp(k,"--hardware"))a.mode=MODE_HARDWARE;
        else if(!strcmp(k,"--mock-hardware")){a.mode=MODE_HARDWARE;a.backend=BACKEND_MOCK;}
        else if(!strcmp(k,"--executor-commit")&&v){a.executor_commit=v;i++;}
        else if(!strcmp(k,"--session-dir")&&v){a.session_dir=v;i++;}else if(!strcmp(k,"--output-dir")&&v){a.output_dir=v;i++;}
        else if(!strcmp(k,"--victim")&&v){a.victim=atoi(v);i++;}else if(!strcmp(k,"--sender")&&v){a.sender=atoi(v);i++;}
        else if(!strcmp(k,"--pin-khz")&&v){a.pin_khz=atol(v);i++;}else if(!strcmp(k,"--slot-s")&&v){a.slot_s=atof(v);i++;}
        else if(!strcmp(k,"--off-window-s")&&v){a.off_window_s=atof(v);i++;}else if(!strcmp(k,"--read-hz")&&v){a.read_hz=atol(v);i++;}
        else if(!strcmp(k,"--temp-veto-c")&&v){a.temp_veto_c=atof(v);i++;}else die("unknown or incomplete option: %s",k);
    }
    if(!a.mode||!a.session_dir||!a.output_dir||a.victim<0||a.sender<0||a.victim==a.sender)die("invalid required arguments");
    if(a.mode==MODE_HARDWARE&&(!a.executor_commit||strlen(a.executor_commit)!=40))die("hardware mode requires --executor-commit");
    if(!shell_safe(a.session_dir)||!shell_safe(a.output_dir))die("unsafe path");
    return a;
}
static void copy_exclusive(const char*s,const char*d){FILE*i=fopen(s,"rb"),*o=fopen(d,"wbx");if(!i||!o)die("copy failed");char b[65536];size_t n;while((n=fread(b,1,sizeof(b),i)))if(fwrite(b,1,n,o)!=n)die("copy failed");fclose(i);fclose(o);}
static Schedule load_schedule(const RunnerArgs*a){
    char p[CP_PATH_MAX],schema[128],manifest_sid[128];
    Schedule s={0};
    path_join(p,sizeof(p),a->session_dir,"session_manifest.json");char*m=slurp(p);
    sha256(p,s.session_manifest_sha256);
    if(jstr(m,"schema_id",schema,sizeof(schema))||strcmp(schema,"CAT_CAS_PHASE6_COMBINED_SESSION_MANIFEST_V1"))die("unexpected session manifest schema");
    if(jstr(m,"session_id",manifest_sid,sizeof(manifest_sid))) die("manifest missing session_id");
    verify_file(m,a->session_dir,"session.json");
    verify_file(m,a->session_dir,"windows.jsonl");
    free(m);
    path_join(p,sizeof(p),a->session_dir,"session.json");char*h=slurp(p);
    if(jstr(h,"schema_id",schema,sizeof(schema))||strcmp(schema,"CAT_CAS_PHASE6_COMBINED_SESSION_SCHEDULE_V1"))die("unexpected session schema");
    if(jstr(h,"session_id",s.session_id,sizeof(s.session_id))||strcmp(s.session_id,manifest_sid))die("session ID mismatch");
    if(jstr(h,"route",s.route,sizeof(s.route))||jstr(h,"campaign_source_commit",s.campaign_source_commit,sizeof(s.campaign_source_commit))||jstr(h,"campaign_plan_sha256",s.campaign_plan_sha256,sizeof(s.campaign_plan_sha256)))die("missing session binding");
    long count;int restoration=1;if(jlong(h,"window_count",&count)||count<=0||jbool(h,"restoration_authorized",&restoration)||restoration)die("invalid session header");free(h);
    s.count=(size_t)count;s.windows=calloc(s.count,sizeof(*s.windows));if(!s.windows)die("oom");
    path_join(p,sizeof(p),a->session_dir,"windows.jsonl");FILE*f=fopen(p,"r");if(!f)die("open windows");char*line=NULL;size_t cap=0,n=0;
    while(getline(&line,&cap,f)!=-1){if(n>=s.count)die("extra schedule rows");Window*w=&s.windows[n];long x;int present;
        if(jlong(line,"window_index",&x)||x!=(long)n) die("window indices not contiguous or duplicate");
        w->window_index=x;
        if(jstr(line,"session_id",w->session_id,sizeof(w->session_id))||strcmp(w->session_id,s.session_id))die("window session ID mismatch");
        if(jstr(line,"stage",w->stage,sizeof(w->stage))||jstr(line,"block_id",w->block_id,sizeof(w->block_id))||jstr(line,"family",w->family,sizeof(w->family))||
           jstr(line,"measurement_mode",w->measurement_mode,sizeof(w->measurement_mode))||jstr(line,"executed_tone_order",w->executed_tone_order,sizeof(w->executed_tone_order))||
           jstr(line,"declared_tone_order",w->declared_tone_order,sizeof(w->declared_tone_order)))die("short schedule row");
        if(jstr(line,"actual_mode",w->actual_mode,sizeof(w->actual_mode))) strcpy(w->actual_mode,"null");
        if(jstr(line,"declared_mode",w->declared_mode,sizeof(w->declared_mode))) strcpy(w->declared_mode,"null");
        if(jbool(line,"drive_on",&w->drive_on)||jbool(line,"sender_off_required",&w->sender_off_required))die("invalid window booleans");
        if(jnullable_long(line,"physical_tone_index",&x,&present)) die("missing physical tone");
        w->physical_tone_index=present?(int)x:-1;
        if(jnullable_long(line,"codeword_source_index",&x,&present)) die("missing codeword source");
        w->codeword_source_index=present?(int)x:-1;
        if(jnullable_long(line,"theta_idx",&x,&present)) die("missing theta_idx");
        w->theta_idx=present?(int)x:-1;
        if(jlong(line,"amplitude_level",&x))w->amplitude_level=w->drive_on?3:0;else w->amplitude_level=(int)x;
        if(w->sender_off_required&&w->drive_on)die("sender_off_required + drive_on rejection");
        if(!strcmp(w->measurement_mode,"raw_ring_sender_off")){if(!w->sender_off_required||w->drive_on)die("raw_ring_sender_off requires sender off");}
        else if(!strcmp(w->measurement_mode,"lockin_and_raw_ring")){if(w->drive_on&&(w->physical_tone_index<0||w->codeword_source_index<0))die("driven lock-in missing physical tone or codeword source");}
        else die("unsupported measurement mode");
        n++;
    }free(line);fclose(f);if(n!=s.count)die("short schedule row count");
    if((!strcmp(s.route,"v4s5")&&(a->victim!=4||a->sender!=5))||(!strcmp(s.route,"v2s3")&&(a->victim!=2||a->sender!=3))) die("invalid route/core pair");
    return s;
}
static void validation_outputs(const RunnerArgs*a,const Schedule*s){
    if(exists(a->output_dir)||mkdir(a->output_dir,0755)) die("refusing existing output directory");
    char x[CP_PATH_MAX],y[CP_PATH_MAX];
    const char*in[]={"session.json","windows.jsonl"};for(int i=0;i<2;i++){path_join(x,sizeof(x),a->session_dir,in[i]);path_join(y,sizeof(y),a->output_dir,in[i]);copy_exclusive(x,y);}
    const char*empty[]={"raw_samples.bin","telemetry.csv","stderr.log"};for(int i=0;i<3;i++){path_join(y,sizeof(y),a->output_dir,empty[i]);FILE*f=fopen(y,"wbx");if(!f)die("create output");fclose(f);}
    path_join(y,sizeof(y),a->output_dir,"stdout.log");FILE*f=fopen(y,"wx");if(!f)die("create stdout.log");fprintf(f,"VALIDATION_ONLY_HARDWARE_NOT_EXECUTED\n");fclose(f);
    path_join(y,sizeof(y),a->output_dir,"window_results.csv");f=fopen(y,"wx");if(!f)die("create window_results.csv");fprintf(f,"window_index,session_id,validation_status,hardware_executed\n");for(size_t i=0;i<s->count;i++)fprintf(f,"%zu,%s,VALIDATED,0\n",i,s->session_id);fclose(f);
    path_join(y,sizeof(y),a->output_dir,"run.json");f=fopen(y,"wx");if(!f)die("create run.json");fprintf(f,"{\n  \"schema_id\": \"CAT_CAS_PHASE6_COMBINED_RUN_V1\",\n  \"session_id\": \"%s\",\n  \"status\": \"VALIDATION_ONLY_HARDWARE_NOT_EXECUTED\",\n  \"hardware_executed\": false,\n  \"automatic_retry\": false,\n  \"restoration_authorized\": false,\n  \"windows_seen\": %zu\n}\n",s->session_id,s->count);fclose(f);
    if(write_run_manifest(a->output_dir,s->session_id,"VALIDATION_ONLY_HARDWARE_NOT_EXECUTED"))die("manifest creation failed");
}
int main(int ac,char**av){RunnerArgs a=parse_args(ac,av);if(!exists(a.session_dir)||exists(a.output_dir))die(exists(a.output_dir)?"refusing existing output directory":"session directory does not exist");Schedule s=load_schedule(&a);int rc=0;if(a.mode==MODE_VALIDATE){validation_outputs(&a,&s);printf("{\"status\":\"VALIDATION_ONLY_HARDWARE_NOT_EXECUTED\",\"session_id\":\"%s\",\"windows\":%zu}\n",s.session_id,s.count);}else rc=run_hardware(&a,&s);free(s.windows);return rc;}
