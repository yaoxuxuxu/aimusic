#include "shazam.cpp"
class shazam_database:Shazam{
    public:
        typedef pair<ull,ull> music;
        
        void insert(bool *cm,ull h,ull t,ull id){
            get_feature(cm,h,t,id);
            sz_vec tmp=get_sflist();
            for(auto i:tmp){
                music tmp;
                tmp.first=i.ta;
                tmp.second=i.id;
                database[i.feature].push_back(tmp);
                //printf("%d %d\n",tmp.first,tmp.second);
            }
            return;
        }
        void count(unsigned int feature){
            for(auto i:database[feature]){
                counter[i.second]++;
            }
            return;
        }
        ull query(bool *cm,ull h,ull t,int top=3){
            counter.clear();
            get_feature(cm,h,t,0);
            sz_vec tmp=get_sflist();
            for(auto i:tmp){
                unsigned int f=i.feature;
                count(f);
            }
            vector<music> rank;
            double sum=0;
            for(auto i:counter){
                sum+=i.second;
                music tmp;
                tmp.first=i.first;
                tmp.second=i.second;
                rank.push_back(tmp);
                //printf("%d,%d\n",i.first,i.second);
            }
            sort(rank.begin(),rank.end(),[this](music xx,music yy){return xx.second>yy.second;});
            for(int i=0;i<rank.size();i++){
                if(i>=top)break;
                printf("id:%d probility:%.2lf%\n",rank[i].first,rank[i].second/sum*100);
            }
            return rank[0].first;
        }
        void clear(){

            database.clear();
        }
    private:
        map<int,int> counter;
        map<unsigned int,vector<music>> database;

};