for i in 3 5 10
do
    python dataset_linear_svm.py --partys $i 
    python launcher.py --world_size $i 
    python launcher_stitch.py --world_size $i
done

for i in 3 5 10
do
    python dataset_linear_svm.py --partys 5 --example $i
    python launcher.py --world_size 5
    python launcher_stitch.py --world_size 5
done