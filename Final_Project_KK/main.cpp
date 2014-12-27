//
//  main.cpp
//  Final Project KK
//
//  Created by Iskandar on 12/11/14.
//  Copyright (c) 2014 Home. All rights reserved.
//
//terbaik sementara 4 hidden node, 01101101, iterasi 18
//Input :
//2
//tested_positive 1
//tested_negative 0
//4  

#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <time.h>
#include <math.h>
#include <vector>
#include <string.h>
#define FITNESS_THRESHOLD 200
#define MAX_ITERATION 100
#define MIN_ITERATION 20
#define LOWER_BOUND 0.0
#define UPPER_BOUND 0.5

using namespace std;

//class untuk menyimpan kelas-kelas yang ada pada MLP
typedef struct my_class
{
    string param;
    double value;
    
} my_class;

//struct node di dalam mlp
typedef struct my_node
{
    string name;
    double value;
    vector<double> weight;
    double error;
    
} node;

//layer dalam mlp, berisi node-node, terdiri dari input, hidden dan output
class layer
{
private:
    vector<node> nodes; //node-node yang berada di layer ini
    int all_nodes; //jumlah semua node yang ada di layer ini
    
public:
    //inisialisasi node dengan memasukkan value, nama dan jumlah semua weight tiap node
    void init_node(string name, double value, int all_weights)
    {
        node my_node;
        my_node.name = name;
        my_node.value = value;
        
        //fungsi set weight dengan metode random
        if(all_weights > 0)
        {
            double lower_bound = LOWER_BOUND;
            double upper_bound = UPPER_BOUND;
            double random_weight;
            
            for(int i = 0; i < all_weights; i++)
            {            
                random_weight = ((double) rand() / RAND_MAX) * (upper_bound - lower_bound) + lower_bound;
                
                //cout << this->all_weights << endl;
                my_node.weight.push_back(random_weight);
            }
        }
        my_node.error = 0;
        
        nodes.push_back(my_node);
    }
    
    //inisialisasi node dengan memasukkan node dan jumlah semua weight tiap node
    void init_node(node node, int all_weights)
    {
        //fungsi set weight dengan metode random
        if(all_weights > 0)
        {
            double lower_bound = LOWER_BOUND;
            double upper_bound = UPPER_BOUND;
            double random_weight;
            
            for(int i = 0; i < all_weights; i++)
            {            
                random_weight = ((double) rand() / RAND_MAX) * (upper_bound - lower_bound) + lower_bound;
                
                //cout << this->all_weights << endl;
                node.weight.push_back(random_weight);
            }
        }
        node.error = 0;
        
        nodes.push_back(node);
    }
    
    //fungsi mendapatkan semua node di layer
    vector<node> get_nodes()
    {
        return this->nodes;
    }
    
    //inisialisasi jumlah semua node yang ada
    void init_all_nodes(int all_nodes)
    {
        this->all_nodes = all_nodes;
    }
    
    //fungsi mendapatkan jumlah semua node yang ada
    int get_all_nodes()
    {
        return this->all_nodes;
    }
    
    //fungsi memasukkan error
    void set_error(int index, double error)
    {
        this->nodes[index].error = error;
    }
    
    //fungsi memasukkan weight
    void set_weight(int node_index, int weight_index, double weight)
    {
        //cout << "dalam fungsi set weight = " << weight;
        this->nodes[node_index].weight[weight_index] = weight;
    }
    
    //fungsi mendapatkan semua weight yang ada pada node tertentu
    vector<double> get_weight(int index)
    {
        return this->nodes[index].weight;
    }
    
    //fungsi menetapkan nilai di node tertentu
    void set_value(int index, double value)
    {
        this->nodes[index].value = value;
    }
    
};

//kelas mlp untuk keseluruhan proses mlp
class MultiLayerPerceptron
{
private:
    vector<layer> layers; //kelas layer yang ada mulai dari input, hidden, dan output
    vector<my_class> classification; //kelas-kelas yang ada
    int layer_size; //jumlah layer di dalam mlp
    int expected_result; //hasil yang seharusnya didapatkan
    double learning_rate; //learning rate
    vector<MultiLayerPerceptron> memory_mlp; //menyimpan mlp terbaik selama proses learning
    vector<double> Max_value; //max value di semua dataset tiap node
    vector<double> Min_value; //min value di semua dataset tiap node
    int fitness_value; //fitness value
    int diabetes_right; //nilai terdapatnya prediksi diabetes yang benar (testing)
    int healthy_right; //nilai terdapatnya prediksi sehat yang benar (testing)
    
    //fungsi memasukkan kelas di dalam fungsi data_training 
    void data_training_class(vector<my_class> my_class)
    {
        for(int i = 0; i < my_class.size(); i++)
        {
            this->classification.push_back(my_class[i]);
        }
    }
    
    //fungsi backprogation
    void backpropagation()
    {
        //dapatkan nilai seharusnya (kelas) dan learning rate
        int expected_result = this->get_expected_result();
        double learning_rate = this->get_learning_rate();
        
        //lakukan sebanyak layer yang ada mulai dari belakang (output)
        for(int i = this->layer_size-1; i >= 0; i--)
        {
            double new_error; //variabel error baru
            
            //dapatkan semua node di layer ini
            vector<node> node_in_this_layer = this->layers[i].get_nodes();
            
            //lakukan sebanyak node yang ada di layer
            for(int j = 0; j < node_in_this_layer.size(); j++)
            {
                //apabila layer merupakan layer output
                if(i == this->layer_size-1)
                {
                    //hitung error dengan rumus z(1-z) * (t-z)
                    new_error = node_in_this_layer[j].value * (1 - node_in_this_layer[j].value) * (expected_result - node_in_this_layer[j].value);
                    
                    set_error_in_layer(i, j, new_error); //set error di layer i, node ke j
                }
                else //apabila layer merupakan hidden node atau input
                {
                    //ambil semua node di layer berikut
                    vector<node> node_in_next_layer = this->layers[i+1].get_nodes();
                    double output_PE = 0; //variabel Output PE (error*weight)
                    
                    //lakukan sebanyak node yang ada di layer berikutnya
                    for (int x = 0; x < node_in_next_layer.size(); x++)
                    {
                        //output pe = error di node next * semua weight yang terhubung
                        output_PE += node_in_next_layer[x].error * node_in_this_layer[j].weight[x];
                        
                        //hitung weight baru (weight baru + (learning rate * semua error output * nilai di node itu
                        double new_weight = node_in_this_layer[j].weight[x] + (learning_rate * node_in_next_layer[x].error * node_in_this_layer[j].value);
                        
                        set_weight_in_layer(i, j, x, new_weight); //masukkan weight baru
                    }
                    
                    //error baru = y(1-y) * Output PE
                    new_error = node_in_this_layer[j].value * (1 - node_in_this_layer[j].value) * output_PE;
                    
                    set_error_in_layer(i, j, new_error); //masukkan error baru
                    node_in_next_layer.clear(); //inisialisasi baru
                }
            }
            node_in_this_layer.clear(); //inisialisasi baru
        }
    }
    
public:
    //inisialisasi learning rate
    void init_learning_rate(double rate)
    {
        this->learning_rate = rate;
    }
    
    //inisialisasi data training
    void init_data_training(int feature, int layer, int output, vector<int> all_hidden)
    {   
        //inisialisasi layer input
        class layer first_layer;    
        first_layer.init_all_nodes(feature);
        this->layers.push_back(first_layer);
        
        //inisialiasi layer hidden
        for(int i = 0; i < all_hidden.size(); i++)
        {
            class layer hidden_layer;
            hidden_layer.init_all_nodes(all_hidden[i]);
            this->layers.push_back(hidden_layer);
        }
        
        //inisialisasi layer output
        class layer output_layer;
        output_layer.init_all_nodes(output);
        this->layers.push_back(output_layer);
        
        //set variabel
        this->layer_size = layer;
        this->fitness_value = 0;
        this->healthy_right = 0;
        this->diabetes_right = 0;
    }
    
    //fungsi mendapatkan nilai learning rate
    double get_learning_rate()
    {
        return this->learning_rate;
    }
    
    //fungsi mendapatkan semua node di layer tertentu
    vector<node> get_node_in_layer(int layer)
    {
        return this->layers[layer].get_nodes();
    }
    
    //mendapatkan node di layer tertentu dengan nama
    node get_node_with_name_in_layer(int layer, string name)
    {
        vector<node> all_nodes = this->get_node_in_layer(layer);
        for(int i = 0; i < all_nodes.size(); i++)
        {
            if(all_nodes[i].name == name)
            {
                return all_nodes[i];
            }
        }
        return all_nodes[-1];
    }
    
    //fungsi mendapatkan node di layer tertentu dengan index
    node get_node_with_index_in_layer(int layer, int index)
    {
        vector<node> all_nodes = this->get_node_in_layer(layer);
        return all_nodes[index];
    }
    
    //fungsi inisialisasi node di layer tertentu dengan input nama node dan nilai
    void init_node_in_layer(int layer, string name, double value, int all_weights)
    {
        this->layers[layer].init_node(name, value, all_weights);
    }
    
    //fungsi inisialisasi node di layer tertentu dengan input node
    void init_node_in_layer(int layer, node node, int all_weights)
    {
        this->layers[layer].init_node(node, all_weights);
    }
    
    //fungsi mendapatkan jumlah layer yang ada
    int get_layer_size()
    {
        return this->layer_size;
    }
    
    //fungsi mendapatkan jumlah semua node di layer tertentu
    int get_all_nodes_in_layer(int layer)
    {
        return this->layers[layer].get_all_nodes();
    }
    
    //fungsi memasukkan nilai error di node tertentu
    void set_error_in_layer(int layer, int index, double error)
    {
        this->layers[layer].set_error(index, error);
    }
    
    //fungsi memasukkan weight di layer dan node tertentu
    void set_weight_in_layer(int layer, int node_index, int weight_index, double weight)
    {
        this->layers[layer].set_weight(node_index, weight_index, weight);
    }
    
    //fungsi mendapatkan semua bobot yang dimiliki node tertentu
    vector<double> get_weight_in_layer(int layer, int index)
    {
        return this->layers[layer].get_weight(index);
    }
    
    //fungsi mendapatkan semua kelas-kelas yang ada
    vector<my_class> get_classification()
    {
        return this->classification;
    }
    
    //fungsi mendapatkan banyaknya kelas yang ada
    int get_classification_size()
    {
        return this->classification.size();
    }
    
    //fungsi memasukkan hasil yang diinginkan
    void set_expected_result(int result)
    {
        this->expected_result = result;
    }
    
    //fungsi mendapatkan hasil yang diinginkan
    int get_expected_result()
    {
        return this->expected_result;
    }
    
    //fungsi mendapatkan fitness value
    int get_fitness_value()
    {
        return this->fitness_value;
    }
    
    //fungsi mendapatkan nilai prediksi diabetes yang benar (testing)
    int get_diabetes_right()
    {
        return this->diabetes_right;
    }
    
    //fungsi mendapatkan nilai prediksi sehat yang benar (testing)
    int get_healthy_right()
    {
        return this->healthy_right;
    }
    
    //fungsi data training
    void data_training(FILE *datasets, int feature, int layer, int output, vector<my_class> class_all, vector<int> all_hidden, string chromosome)
    {   
        //inisialisasi awal
        this->fitness_value = 0; 
        this->healthy_right = 0;
        this->diabetes_right = 0;
        
        vector< vector<node> > all_data; //data semua input
        vector<my_class> all_output; //data semua kelas prediksi output
        
        char data[100]; //data berupa string yang dimasukkan
        char *tmp; //pembacaan string
        
        //pembacaan datasets.txt
        try 
        {
            while(fgets(data, 100, datasets) != NULL) //pembacaan datasets
            {   
                //inisialisasi tiap node
                int count = 1;
                vector<node> input;
                tmp = strtok(data, ",");  //pembacaan tiap node dipisah oleh tanda koma (,)
                while(tmp != NULL)
                {   
                    //inisialisasi nama tiap node
                    char index[3];
                    string name = "input_node_";
                    sprintf(index, "%d", count);
                    name = name + index;
                    node data_now;
                    
                    //inisialisasi value tiap node apabila number
                    if(isnumber(*tmp))
                    {
                        //inisialisasi value node
                        data_now.name = name;
                        data_now.value = atof(tmp);
                        input.push_back(data_now);
                    }
                    else //inisialisasi value apabila berupa kelas
                    {
                        my_class output;
                        
                        for(int x = 0; x < class_all.size(); x++)
                        {
                            if(tmp == class_all[x].param + "\n")
                            {
                                //perubahan value dari kelas ke angka
                                output.param = tmp;
                                output.value = class_all[x].value;
                                all_output.push_back(output);
                            }
                        }
                    }
                    tmp = strtok(NULL, ",");
                    count++;
                }
                all_data.push_back(input); //masukkan data
                input.clear();
            }
        }
        //exception apabila terjadi kesalahan dalam pembacaan dataset
        catch (exception e) 
        {
            cout << e.what() << endl;
        }
        
        //pencarian nilai max dan min di semua dataset dibagi tiap node yang ada
        vector<double> max_value_in_node;
        vector<double> min_value_in_node;
        double max = -9999;
        double min = 9999;
        
        for(int i = 0; i < feature; i++)
        {
            for(int j = 0; j < all_data.size(); j++)
            {
                if(all_data[j][i].value >= max)
                {
                    max = all_data[j][i].value;
                }
                
                if(all_data[j][i].value <= min)
                {
                    min = all_data[j][i].value;
                }
            }
            max_value_in_node.push_back(max);
            min_value_in_node.push_back(min);
        }
        
        this->Max_value = max_value_in_node;
        this->Min_value = min_value_in_node;
        
        //memulai data training        
        vector<MultiLayerPerceptron> memory_mlp;
        int coba = 0; //variable coba untuk mengetahui sudah berapa kali proses mlp terjadi
        
        //sejumlah semua data yang ada lakukan proses mlp
        for(int i = 0; i < all_data.size(); i++)
        {
            vector<node> data_input = all_data[i]; //masukkan data satu mlp ke data_input
            
            MultiLayerPerceptron training_mlp; //kelas mlp untuk dijadikan objek training
            
            training_mlp.data_training_class(class_all); //memasukkan semua kelas yang ada
            training_mlp.init_data_training(feature, layer, output, all_hidden); //inisialisasi
            training_mlp.set_expected_result(all_output[i].value); //memasukkan hasil seharusnya
            training_mlp.init_learning_rate(0.1); //inisialisasi learning rate
            coba++; // satu mlp telah dibuat
            
            //pembacaan tiap node input
            for(int j = 0; j < data_input.size(); j++)
            {
                if(i == 0) //awal inisialisasi
                {
                    if(data_input[j].value != 0 || data_input[j].value != 1) //apabila data bukan kelas
                    {
                        //cocokkan kromosom
                        if(chromosome[j] == '0')
                        {
                            data_input[j].value = 0;
                        }
                        
                        //normalisasi data
                        data_input[j].value = (data_input[j].value - min_value_in_node[j]) /  (max_value_in_node[j] - min_value_in_node[j]);
                        
                        //apabila node dalam mlp belum diinisialisasi
                        if(training_mlp.get_node_in_layer(0).size() < training_mlp.get_all_nodes_in_layer(0))
                        {
                            //inisialisasi node dengan random weight
                            training_mlp.init_node_in_layer(0, data_input[j], training_mlp.get_all_nodes_in_layer(1));
                        }
                        else // apabila sudah terdapat node dalam mlp
                            if(training_mlp.get_node_in_layer(0).size() == training_mlp.get_all_nodes_in_layer(0))
                            {
                                //set value ke node yang sudah ada dan memiliki weight
                                training_mlp.set_value_in_layer(0, j, data_input[j].value);
                            }
                        
                        //apabila terdapat mlp terbaik sebelumnya
                        if(memory_mlp.size() > 0)
                        {
                            //gunakan weight terbaik sebelum yang sudah mengalami training
                            vector<double> weight = memory_mlp.back().get_weight_in_layer(0, j);
                            
                            for(int x = 0; x < weight.size(); x++)
                            {
                                training_mlp.set_weight_in_layer(0, j, x, weight[x]);
                            }
                            
                            weight.clear();
                        }
                    }
                    else //apabila data berupa kelas
                    {
                        //cocokkan dengan kromosom
                        if(chromosome[j] == '0')
                        {
                            data_input[j].value = 0;
                        }
                        
                        //apabila node di mlp belum diinisialisasi
                        if(training_mlp.get_node_in_layer(0).size() < training_mlp.get_all_nodes_in_layer(0))
                        {
                            //inisialisasi node
                            training_mlp.init_node_in_layer(0, data_input[j], training_mlp.get_all_nodes_in_layer(1));
                        }
                        else //apabila sudah terdapat node dalam mlp
                            if(training_mlp.get_node_in_layer(0).size() == training_mlp.get_all_nodes_in_layer(0))
                            {
                                //set value ke node yang sudah ada dan memiliki weight
                                training_mlp.set_value_in_layer(0, j, data_input[j].value);
                            }
                        
                        //apabila terdapat mlp terbaik sebelumnya 
                        if(memory_mlp.size() > 0)
                        {
                            //gunakan weight terbaik sebelum yang sudah mengalami training
                            vector<double> weight = memory_mlp.back().get_weight_in_layer(0, j);
                            
                            for(int x = 0; x < weight.size(); x++)
                            {
                                training_mlp.set_weight_in_layer(0, j, x, weight[x]);
                            }
                            
                            weight.clear();
                        }
                    }
                }
                else //apabila bukan data pertama
                {
                    //apabila data bukan kelas
                    if(data_input[j].value != 0 || data_input[j].value != 1)
                    {
                        //cocokkan dengan kromosom
                        if(chromosome[j] == '0')
                        {
                            data_input[j].value = 0;
                        }
                        
                        //normalisasi data
                        data_input[j].value = (data_input[j].value - min_value_in_node[j]) /  (max_value_in_node[j] - min_value_in_node[j]);
                        
                        //apabila belum ada node
                        if(training_mlp.get_node_in_layer(0).size() < training_mlp.get_all_nodes_in_layer(0))
                        {
                            //inisialisasi node
                            training_mlp.init_node_in_layer(0, data_input[j], training_mlp.get_all_nodes_in_layer(1));
                        }
                        else //apabila sudah ada node
                            if(training_mlp.get_node_in_layer(0).size() == training_mlp.get_all_nodes_in_layer(0))
                            {
                                //set value di node
                                training_mlp.set_value_in_layer(0, j, data_input[j].value);
                            }
                        
                        //apabila terdapat data mlp terbaik sebelumnya
                        if(memory_mlp.size() > 0)
                        {
                            //gunakan weight terbaik sebelum yang sudah mengalami training
                            vector<double> weight = memory_mlp.back().get_weight_in_layer(0, j);
                            
                            for(int x = 0; x < weight.size(); x++)
                            {
                                training_mlp.set_weight_in_layer(0, j, x, weight[x]);
                            }
                            
                            weight.clear();
                        }
                    }
                    else //apabila berupa kelas
                    {
                        //cocokkan dengan kromosom
                        if(chromosome[j] == '0')
                        {
                            data_input[j].value = 0;
                        }
                        
                        //apabila belum ada node
                        if(training_mlp.get_node_in_layer(0).size() < training_mlp.get_all_nodes_in_layer(0))
                        {
                            //inisialisasi node
                            training_mlp.init_node_in_layer(0, data_input[j], training_mlp.get_all_nodes_in_layer(1));
                        }
                        else //apabila sudah ada node
                            if(training_mlp.get_node_in_layer(0).size() == training_mlp.get_all_nodes_in_layer(0))
                            {
                                //set value di node
                                training_mlp.set_value_in_layer(0, j, data_input[j].value);
                            }
                        
                        //apabila terdapat data mlp terbaik sebelumnya
                        if(memory_mlp.size() > 0)
                        {
                            //gunakan weight terbaik sebelum yang sudah mengalami training
                            vector<double> weight = memory_mlp.back().get_weight_in_layer(0, j);
                            
                            for(int x = 0; x < weight.size(); x++)
                            {
                                training_mlp.set_weight_in_layer(0, j, x, weight[x]);
                            }
                            
                            weight.clear();
                        }
                        
                    }
                    
                }
            }
            
            //masukkan data mlp ke data mlp terbaik saat ini
            training_mlp.memory_mlp.push_back(training_mlp);
            
            //lakukan proses feed forward
            training_mlp.start_process("data_training"); 
            
            //apabila sudah melebihi data ke 400, lakukan latihan prediksi
            if(coba > 400)
            {
                //hitung fitness value dan kebenaran data
                this->fitness_value += training_mlp.get_fitness_value();
                this->healthy_right += training_mlp.get_healthy_right();
                this->diabetes_right += training_mlp.get_diabetes_right();
            }
            
            //jika data belum mencapai data ke 400, lakukan perbaikan error (backprop)
            if(coba <= 400)
            {
                //lakukan backpropagation
                training_mlp.backpropagation();
            }
            
            //masukkan data mlp sekarang ke data mlp terbaik
            memory_mlp.push_back(training_mlp); 
            
            //apabila data memory lebih dari satu keluarkan yang sudah ada, masukkan yang baru
            if(memory_mlp.size() > 1)
            {
                MultiLayerPerceptron tmp = memory_mlp.back();
                while (memory_mlp.size() > 0)
                {
                    memory_mlp.pop_back();
                }
                memory_mlp.push_back(tmp);
            }
            
            //cout << "----End training data----" << endl << endl; 
        }
        
        this->memory_mlp.push_back(memory_mlp.back()); //masukkan ke data memory kelas ini
        
    }
    
    //fungsi feed forward
    void start_process(string process)
    {
        int layer_size = this->get_layer_size(); //ambil jumlah semua layer
        
        //lakukan sebanyak layer yang ada -1
        for(int i = 0; i < layer_size-1; i++)
        {
            class layer this_layer = this->layers[i]; //set variable layer sekarang
            class layer next_layer = this->layers[i+1]; //set variable layer berikutnya
            class layer old_layer = this->memory_mlp.back().layers[i]; //set variable mlp terbaik sebelum
            
            //ambil semua node dan jumlah node di layer saat ini
            int this_layer_all_nodes = this_layer.get_all_nodes();
            vector<node> this_layer_nodes = this_layer.get_nodes();
            
            //ambil semua node dan jumlah node di layer berikutnya
            int next_layer_all_nodes = next_layer.get_all_nodes();
            vector<node> next_layer_nodes = next_layer.get_nodes();
            
            //ambil semua node di mlp hasil training terakhir
            vector<node> old_layer_nodes = old_layer.get_nodes();
            
            //lakukan sebanyak node di layer berikutnya
            for(int x = 0; x < next_layer_all_nodes; x++)
            {
                double dot_product_all = 0; //inisialisasi dot product
                
                //lakukan sebanyak node di layer saat ini
                for(int y = 0; y < this_layer_all_nodes; y++)
                {
                    //hitung dot product di weight node saat ini
                    double dot_product = this_layer_nodes[y].value * old_layer_nodes[y].weight[x];
                    
                    //semua dot product dari weight yang terhubung ke node berikutnya
                    dot_product_all += dot_product;
                }
                
                double exp_value = exp(-dot_product_all); //nilai eksponen
                
                double sigmoid_value = 1 / (1 + exp_value); //nilai fungsi sigmoid 
                
                string name = "";
                
                // apabila sekarang bukan merupakan layer terakhir kedua
                if(i < layer_size-2)
                {
                    //apabila node di layer berikutnya belum ada
                    if(next_layer.get_nodes().size() < next_layer.get_all_nodes())
                    {
                        //inisialisasi node
                        name = "hidden_node_";
                        char index[3];
                        sprintf(index, "%d", x+1);
                        name = name + index;
                        
                        //inisialisasi dengan menggunakan sigmoid value
                        next_layer.init_node(name, sigmoid_value, this->layers[i+2].get_all_nodes());
                    }
                    else //apabila sudah ada node di layer berikutnya
                        if(next_layer.get_nodes().size() == next_layer.get_all_nodes())
                        {
                            //set value di node tersebut
                            next_layer.set_value(x, sigmoid_value);
                        }
                }
                else // apabila sekarang adalah layer terakhir kedua (layer sebelum output)
                    if(i == layer_size-2) 
                    {
                        //apabila belum ada node di layer berikutnya
                        if(next_layer.get_nodes().size() < next_layer.get_all_nodes())
                        {
                            //inisialisasi node
                            name = "output_node_";
                            char index[3];
                            sprintf(index, "%d", x+1);
                            name = name + index;
                            
                            //inisialisasi dengan menggunakan dot product
                            next_layer.init_node(name, dot_product_all, 0);
                        }
                        else //apabila sudah ada node di layer berikutnya
                            if(next_layer.get_nodes().size() == next_layer.get_all_nodes())
                            {
                                //set value di layer berikutnya dengan dot product
                                next_layer.set_value(x, dot_product_all);
                            }
                    }
                    else //apabila error
                    {
                        name = " ";
                        cout << "error index" << endl;
                    }
            }
            
            //masukkan layer hasil hitungan ke layer yang dimaksud dan masukkan ke memory juga
            this->layers[i+1] = next_layer;
            this->memory_mlp.back().layers[i+1] = next_layer;
            
            //jika sudah merupakan layer terakhir kedua (layer sebelum output)
            if(i == layer_size - 2)
            {
                //ambil node di output
                vector<node> temp = this->layers[layer_size-1].get_nodes();
                for(int x = 0; x < temp.size(); x++)
                {
                    if(temp[x].value > 0.5) //apabila hasil melebihi threshold
                    {
                        //keluarkan prediksi diabetes apabila process merupakan prediksi
                        if(process == "prediction")
                        {
                            cout << "You get Diabetes" << endl;
                            cout << "Value = " << temp[x].value << endl;
                        }
                        
                        //apabila hasil yang dimaksud positive, maka masukkan nilai fitness value dan diabetes
                        if(this->get_expected_result() == 1)
                        {
                            this->fitness_value++;
                            this->diabetes_right++;
                            //cout << "Diabetes right answer" << endl;
                        }
                    }
                    else //apabila data kurang dari threshold
                    {
                        //keluarkan prediksi sehat apabila process merupakan prediksi
                        if(process == "prediction")
                        {   
                            cout << "You are healthy, Congrats!" << endl;
                            cout << "Value = " << temp[x].value << endl;
                        }
                        
                        //apabila hasil yang dimaksud negative, maka masukkan nilai fitness value dan sehat
                        if(this->get_expected_result() == 0)
                        {
                            this->fitness_value++;
                            this->healthy_right++;
                            //cout << "Healthy right answer" << endl;
                            
                        }
                    }
                    temp.clear(); //temp clear
                }
            }
            
            this_layer_nodes.clear();
            next_layer_nodes.clear();
            old_layer_nodes.clear();
            //pembersihan variabel 
        } 
    }
    
    //fungsi masukkan nilai di node dan layer tertentu
    void set_value_in_layer(int layer, int index, double value)
    {
        this->layers[layer].set_value(index, value);
    }
    
    //fungsi prediksi terdiri dari input dan kromosom hasil GA
    void prediction(vector<double> input, string best_chromosome)
    {
        MultiLayerPerceptron calculate = this->memory_mlp.back(); //ambil mlp terbaik hasil training
        
        //lakukan sebanyak input / node
        for(int i = 0; i < input.size(); i++)
        {
            //apabila data bukan kelas
            if(input[i] != 0 || input[i] != 1)
            {
                //normalisasi data
                input[i] = (input[i] - this->Min_value[i]) / (this->Max_value[i] - this->Min_value[i]);
                
                //cocokkan dengan kromosom
                if(best_chromosome[i] == '0')
                {
                    input[i] = 0;
                }
                
                //masukkan nilai di node
                calculate.set_value_in_layer(0, i, input[i]);
            }
            else //apabila data merupakan kelas
            {
                //cocokkan dengan kromosom
                if(best_chromosome[i] == '0')
                {
                    input[i] = 0;
                }
                
                //masukkan nilai di node
                calculate.set_value_in_layer(0, i, input[i]);
            }
        }
        calculate.start_process("prediction"); //mulai prediksi
    }
};

//kelas Genetic Algorithm
class GA
{
private:
    vector<MultiLayerPerceptron> mlp; //MLP yang berada di GA
    vector<string> chromosome; //kromosom yang dimiliki
    vector<int> fitness_value; //fitness value tiap mlp
    MultiLayerPerceptron best_mlp; //mlp terbaik dari hasil GA
    string best_chromosome; //kromosom terbaik dari hasil GA
    static const int all_mlp = 8; //jumlah MLP yang di proses oleh GA (static)
    
    //fungsi inisialisasi kromosom
    void init_chromosome()
    {
        int int_string; //integer yang akan di random
        
        int count = this->all_mlp; //ambil jumlah semua mlp yang ikut dalam proses GA
        
        //lakukan sebanyak jumlah semua mlp dalam proses GA
        for(int x = 0; x < count; x++)
        {
            string bit_string; //string
            
            for(int i = 0; i < 8; i++)
            {
                int_string = rand() % 2; //random angka 0 atau 1
                char binary[2] = {'0', '1'}; //array berisi karakter 0 atau 1
                bit_string += binary[int_string]; //masukkan isi karekter tersebut
            }
            
            //masukkan kromosom
            this->chromosome.push_back(bit_string);
        }
    }
    
    //fungsi one point cross over
    void cross_over(int index1, int index2, int feature)
    {
        int rand_point = rand() % feature; //nilai random index dari 0 sampai jumlah mlp dalam proses
        string temp; //string untuk proses swap
        
        //lakukan sebanyak input
        for(int i = rand_point; i < feature; i++)
        {
            //proses swapping dari index ke i sampai jumlah fitur kromosom 1 dengan kromosom 2
            temp = this->chromosome[index1][i];
            this->chromosome[index1][i] = this->chromosome[index2][i];
            this->chromosome[index2][i] = temp[0];
        }
    }
    
public:
    
    //fungsi inisialisasi GA terdiri dari feature yang ada, layer dan output
    void init_GA(int feature, int layer, int output)
    {
        int all_class; //jumlah total kelas-kelas yang dimasukkan
        string param; //parameter kelas
        double value; //nilai dari kelas
        vector<my_class> class_all; //semua kelas yang ada 
        my_class input_class; //variabel temp menyimpan input
        
        //proses input oleh user memasukkan kelas-kelas yang ada
        cout << "Jumlah kelas output dalam data: ";
        cin >> all_class;
        
        if(all_class > 0)
        {
            for(int i = 0; i < all_class; i++)
            {
                cout << "Parameter kelas dan nilai dipisah spasi (cth: kotak 1): ";
                cin >> param;
                cin >> value;
                
                if(param != " " && value >= 0 && value <= 1)
                {
                    input_class.param = param;
                    input_class.value = value;
                    class_all.push_back(input_class);
                }
                else {
                    cout << "Error dalam memasukkan data" << endl;
                }
            }
        }
        //error apabila tidak ada kelas
        else {
            cout << "Kelas harus ada!" <<endl;
            exit(1);
        }
        
        vector<int> all_hidden; //jumlah node-node di hidden layer
        //input jumlah node tiap hidden layer
        for(int i = 1; i < layer-1; i++)
        {
            int tmpint;
            cout << "Jumlah node di hidden layer " << i << ": ";
            cin >> tmpint;
            all_hidden.push_back(tmpint);
        }
        
        vector<int> all_fitness_value; //semua data fitness value
        this->init_chromosome(); //inisialisasi kromosom
        
        vector<MultiLayerPerceptron> temp; //temporary
        
        MultiLayerPerceptron new_mlp; //mlp yang digunakan dalam proses GA
        
        //kromosom 111111 sesuai jumlah fitur untuk inisialisasi awal mlp (gunakan semua fitur)
        string all_use; 
        for(int i = 0; i < feature; i++)
        {
            all_use += '1';
        }
        
        //training mlp dengan data menggunakan kromosom semua fitur digunakan
        FILE *training_data;
        training_data = fopen("/Users/deenna/Data/Data ilmu dan informasi/Data Codingan Xcode/FP KK in C/Final Project KK revisi/Final Project KK revisi/Datasets.txt", "r");
        
        new_mlp.data_training(training_data, feature, layer, output, class_all, all_hidden, all_use);
        
        //buat mlp sama sebanyak jumlah mlp yang diproses GA
        for(int i = 0; i < this->all_mlp; i++)
        {
            temp.push_back(new_mlp);
        }
        
        fclose(training_data); // tutup file
        
        //lakukan training data program
        cout << "----Training with Data----" << endl;
        
        for(int y = 0; y < MAX_ITERATION; y++) //lakukan sebanyak max iterasi
        {
            for(int i = 0; i < this->all_mlp; i++) //lakukan sebanyak mlp yang diproses GA
            {
                if(y == 0) //apabila iterasi awal
                {                    
                    //lakukan training dengan mlp yang sama semua tapi dengan kromosom berbeda
                    FILE *training_data;
                    training_data = fopen("/Users/deenna/Data/Data ilmu dan informasi/Data Codingan Xcode/FP KK in C/Final Project KK revisi/Final Project KK revisi/Datasets.txt", "r");
                    
                    //gunakan kromosom untuk training
                    temp[i].data_training(training_data, feature, layer, output, class_all, all_hidden, this->chromosome[i]);
                    
                    //testing (bisa dihapus)
                    cout << "Healthy Right mlp ke " << i << " = " << temp[i].get_healthy_right() << endl;
                    cout << "Diabetes Right mlp ke " << i << " = " << temp[i].get_diabetes_right() << endl << endl;
                    
                    //masukkan nilai fitness value dan mlp hasil training dengan kromosom
                    this->fitness_value.push_back(temp[i].get_fitness_value());
                    this->mlp.push_back(temp[i]);
                    
                    //masukkan data fitness value
                    all_fitness_value.push_back(temp[i].get_fitness_value());
                    
                    //tutup data
                    fclose(training_data);
                    
                    //hapus temp jika tidak digunakan
                    if(i == this->all_mlp-1)
                    {
                        temp.clear();
                    }
                }
                //apabila bukan iterasi awal
                else
                {
                    //lakukan training data
                    FILE *training_data;
                    training_data = fopen("/Users/deenna/Data/Data ilmu dan informasi/Data Codingan Xcode/FP KK in C/Final Project KK revisi/Final Project KK revisi/Datasets.txt", "r");
                    
                    //gunakan kromosom untuk training
                    this->mlp[i].data_training(training_data, feature, layer, output, class_all, all_hidden, this->chromosome[i]);
                    
                    //testing (bisa dihapus)
                    cout << "Healthy Right mlp ke " << i << " = " << this->mlp[i].get_healthy_right() << endl;
                    cout << "Diabetes Right mlp ke " << i << " = " << this->mlp[i].get_diabetes_right() << endl << endl;
                    
                    //masukkan databaru
                    this->fitness_value.push_back(this->mlp[i].get_fitness_value());
                    temp.push_back(this->mlp[i]);
                    
                    all_fitness_value.push_back(this->mlp[i].get_fitness_value());
                    
                    //apabila sudah sampai data akhir, inisialisasi vector baru lagi
                    if(temp.size() == feature)
                    {
                        this->mlp.clear();
                        this->mlp = temp;
                        temp.clear();
                    }
                    
                    //tutup data
                    fclose(training_data);
                    
                }
            }
            
            //inisialisasi nilai max, max pertama, dan max kedua
            int max_now = 0;
            int index_max1 = -1;
            int index_max2 = -1;
            
            //lakukan sebanyak nilai fitness value yang ada dari masing-masing mlp
            for(int i = 0; i < all_fitness_value.size(); i++)
            {
                //testing (bisa dihapus)
                cout << "Fitness value mlp ke - " << i << " = " << all_fitness_value[i] << endl;
                
                //set nilai max sekarang dan ambil max index pertama nya
                if(max_now <= all_fitness_value[i])
                {
                    max_now = all_fitness_value[i];
                    index_max1 = i;
                }
            }
            
            cout << endl;
            max_now = 0; //inisialisasi dari awal lagi
            
            for(int i = 0; i < all_fitness_value.size(); i++)
            {
                //set nilai max sekarang
                if(max_now <= all_fitness_value[i])
                {
                    //apabila indexnya tidak sama dengan max index pertama
                    if(index_max1 != i)
                    {
                        //ambil index max
                        max_now = all_fitness_value[i];
                        index_max2 = i;
                    }
                }
            }
            
            //apabila sudah mencapai threshold atau mencapai max iterasi
            if((all_fitness_value[index_max1] > FITNESS_THRESHOLD || y == MAX_ITERATION-1) && y > MIN_ITERATION)
            {
                //masukkan best mlp data mlp dengan index paling tinggi
                this->best_mlp = this->mlp[index_max1];
                
                //masukkan best chromosome dengan kromosom terbaik
                this->best_chromosome = this->chromosome[index_max1];
                
                cout << "Iterasi ke - " << y << endl;
                //testing (bisa dihapus)
                cout << "chromosome data yang digunakan = " << this->chromosome[index_max1] << endl;
                break;
            }
            
            this->cross_over(index_max1, index_max2, feature); //lakukan cross over max 1 dan max 2
            
            all_fitness_value.clear(); //inisialisasi ulang
        }
        
        cout << "----End training data----" << endl << endl; 
        //akhir data training
        
    }
    
    //fungsi prediksi dengan GA
    void with_GA_prediction(vector<double> input)
    {
        //ambil mlp terbaik dari proses GA
        MultiLayerPerceptron mlp_predict = this->best_mlp;
        
        //lakukan prediksi
        mlp_predict.prediction(input, this->best_chromosome);
    }
};

//fungsi main
int main() 
{   
    GA my_GA; //GA yang akan dilakukan
    vector<double> input; //semua input
    double tmp_input; //input satu-satu
    srand(time(NULL)); //srand untuk rand agar berbeda
    
    my_GA.init_GA(8, 3, 1); //inisialisasi GA dengan 8 fitur, 3 layer dan 1 output
    char confirm = 'y'; //konfirmasi karakter
    
    //perulangan
    while(confirm == 'y')
    {
        //lakukan input sebanyak fitur
        for(int i = 0; i < 8; i++)
        {
            cout << "Masukkan Data ke " << i+1 << " -> ";
            cin >> tmp_input;
            input.push_back(tmp_input);
        }
        my_GA.with_GA_prediction(input); //lakukan prediksi GA
        cout << "lanjutkan? (y/n) -> "; //masukkan pilihan
        cin >> confirm;
        input.clear();
    }
    int halt; //halt
    cin >> halt; //halt
    return 0;
}
