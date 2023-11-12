
```bash
wc -l train_data.csv #查看行总数
head -10 train_data.csv #查看前10行
cat train_data.csv | cut -f2,4 -d"," | more # 查看第2列到第4列，分隔符为逗号
split -l 10000 -d -a3 train_data.csv train0 #切割文件，每个文件1万行，输出为train0xxx
```
