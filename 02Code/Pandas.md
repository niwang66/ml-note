#  华映量云

输出每个地区销售最好的产品名称和总销售量，以及销售最好的产品的购买者姓名和购买量
```csv
region,customer,product,quantity,price
Beijing,A,P1,2,100
Beijing,A,P2,4,200
Shanghai,B,P1,1,100
Shanghai,C,P2,3,200
Beijing,B,P3,2,150
Shanghai,C,P3,1,150
```


``` python
df = pd.read_csv('data.csv')
cities = df['region'].drop_duplicates()
s = df.groupby(['region','product'])['quantity'].sum()
best_prods = []
for city in cities:
    best_prod_in_city = s[(city)].idxmax()
    best_prods.append(best_prod_in_city)
    best_prod_quantity_in_city = s[(city)].max()

print("%s best product: %s, quantity: %s" % (city, best_prod_in_city, best_prod_quantity_in_city))
print(df[df['product'].isin(best_prods)])
```