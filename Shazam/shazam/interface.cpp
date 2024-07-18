#include "database.cpp"
extern "C"{
    shazam_database* sz=new shazam_database;
    void insert(bool *cm,int h,int t,int id){
        sz->insert(cm,h,t,id);
        return;
    }
    void query(bool *cm,int h,int t){
        sz->query(cm,h,t);
        return;
    } 
}
