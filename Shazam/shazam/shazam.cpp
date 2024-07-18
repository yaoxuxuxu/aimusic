#include <iostream>
#include <vector>
#include <map>
#include <algorithm>
#define ull unsigned long long
using namespace std;
class Shazam{
    public:
        struct cm_feature{
            ull h=0;ull t=0;
        };
        struct shazam_feature_before_hash{
            ull fa=0,fp=0,dt=0;
            ull ta=0,id=0;
        };
        
        struct shazam_feature{
            unsigned int feature=0;
            ull id=0,ta=0;
        };
        typedef vector<cm_feature> cm_vec;
        typedef vector<shazam_feature> sz_vec;

        sz_vec get_sflist(){
            return sf_list;
        }
        void get_feature(bool *cm,ull h,ull t,ull id=0){
            cm_list.clear();
            sf_list.clear();
            turn_cm_into_pair(cm,h,t);
            grouping_and_hashing(id);
        }
        unsigned int hash_feature(ull fa,ull fp,ull dt){
            //total 32bit
            //using 9 bit for fa and fp
            //using 14 bit for dt
            //fa|fp|dt
            unsigned int feature=dt+(fp<<14)+(fa<<(14+9));
            return feature;
        }
        void grouping_and_hashing(int id,int group_size=5,int overlap=2,int anchor_offset=-1){
            int len=cm_list.size();
            vector <shazam_feature_before_hash> sfbh;
            sfbh.clear();
            for(ull l=0,r=group_size-1;r<len;){
                ull anchor=anchor_offset+l;
                if(anchor>0){
                    for(int i=l;i<=r;i++){
                        shazam_feature_before_hash tmp;
                        tmp.fa=cm_list[anchor].h;
                        tmp.fp=cm_list[i].h;
                        tmp.ta=cm_list[anchor].t;
                        tmp.dt=cm_list[i].t-tmp.ta;
                        tmp.id=id;
                        sfbh.push_back(tmp);
                    }
                    //printf("%d %d\n",l,r);
                }
                l=l+group_size-overlap;
                r=l+group_size-1;
            }
            
            for(auto i:sfbh){
                shazam_feature tmp;
                tmp.feature=hash_feature(i.fa,i.fp,i.dt);
                tmp.id=i.id;
                tmp.ta=i.ta;
                sf_list.push_back(tmp);
            }
            return;
        }
    private:
        cm_vec cm_list;
        sz_vec sf_list;

        ull get_pos_from_cm(ull x,ull y,ull t){
            return t*x+y;
        }
        void turn_cm_into_pair(bool *cm_tmp,ull h,ull t){
            for(ull i=0;i<h;i++){
                for(ull j=0;j<t;j++){
                    if(cm_tmp[get_pos_from_cm(i,j,t)]){
                        cm_feature tmp;
                        tmp.h=i;
                        tmp.t=j;
                        cm_list.push_back(tmp);
                    }
                }
            }
            sort(cm_list.begin(),cm_list.end(),[this](cm_feature a,cm_feature b){return cm_feature_cmp(a,b);});
            return;
        }
        bool cm_feature_cmp(cm_feature xx,cm_feature yy){
            if(xx.t==yy.t)
                return xx.h<yy.h;
            return xx.t<yy.t; 
        }
        
};