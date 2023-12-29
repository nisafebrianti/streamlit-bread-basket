import streamlit as st
import pandas as pd
import numpy as np
from mlxtend.frequent_patterns import association_rules, apriori

#load dataset
df = pd.read_csv('BreadBasket_DMS.csv')

df['Date']=pd.to_datetime(df['Date'], format = "%Y-%m-%d")

df['month']=df['Date'].dt.month
df['day']=df['Date'].dt.weekday

df["month"].replace([i for  i in range(1,12 + 1)], ["Januari", "February", "Maret", "April", "Mei", "Juni", "Juli", "Agustus", "September", "Oktober", "November", "Desember"], inplace=True)
df["day"].replace([i for i in range(6+1)], ["Senin", "Selasa", "Rabu", "Kamis", "Jumat", "Sabtu", "Minggu"], inplace=True)

st.title("Market Basket Analisis Apriori")

def get_data(month = '', day = ''):
    data = df.copy()
    filtered = data.loc[
        (data["month"].str.contains(month.title())) &
        (data["day"].str.contains(day.title()))
    ]
    return filtered if filtered.shape[0] else "No Data"

def user_input_features():
    item = st.selectbox("Item",['Alfajores', 'Hot chocolate', 'Cookies', 'Muffin', 'Coffee', 'Pastry', 'Medialuna', 'Tea', 'none','Tartine', 'Basket', 'Mineral water', 'Farm house', 'Fudge','Juice', "Ella's kitchen pouches", 'Victorian sponge', 'Frittata','Hearty & seasonal', 'Soup', 'Pick and mix bowls', 'Smoothies','Cake', 'Mighty protein', 'Chicken sand', 'Coke','My-5 fruit shoot', 'Focaccia', 'Sandwich', 'Alfajores', 'Eggs','Brownie', 'Dulce de leche', 'Honey', 'The bart', 'Granola','Fairy doors', 'Empanadas', 'Keeping it local', 'Art tray','Bowl nic pitt', 'Bread pudding', 'Adjustment', 'Truffles','Chimichurri oil', 'Bacon', 'Spread', 'Kids biscuit', 'Siblings','Caramel bites', 'Jammie dodgers', 'Tiffin', 'Olum & polenta','Polenta', 'The nomad', 'Hack the stack', 'Bakewell','Lemon and coconut', 'Toast', 'Scone', 'Crepes', 'Vegan mincepie','Bare popcorn', 'Muesli', 'Crisps', 'Pintxos', 'Gingerbread syrup','Panatone', 'Brioche and salami', 'Afternoon with the baker','Salad', 'Chicken stew', 'Spanish brunch','Raspberry shortbread sandwich', 'Extra salami or feta','Duck egg', 'Baguette', "Valentine's card", 'Tshirt','Vegan feast', 'Postcard', 'Nomad bag', 'Chocolates','Coffee granules', 'Drinking chocolate spoons', 'Christmas common','Argentina night', 'Half slice monster', 'Gift voucher','Cherry me dried fruit', 'Mortimer', 'Raw bars', 'Tacos/fajita'])
    day = st.selectbox("Day",["Senin", "Selasa", "Rabu", "Kamis", "Jumat", "Sabtu", "Minggu"])
    month = st.select_slider("Month",["Januari", "February", "Maret", "April", "Mei", "Juni", "Juli", "Agustus", "September", "Oktober", "November", "Desember"], value="Januari")

    return day,month, item

day,month,item = user_input_features()

data = get_data(month, day)

def encode (x):
    if x <= 0:
        return 0
    elif x>=1:
        return 1

if type(data) != type ("No Result"):
    item_count = data.groupby(["Transaction","Item"])["Item"].count().reset_index(name="Count")
    item_count_pivot = item_count.pivot_table(index='Transaction', columns= 'Item', values = 'Count', aggfunc = 'sum').fillna(0)
    item_count_pivot = item_count_pivot.applymap(encode)
    
    support = 0.01
    frequent_items = apriori(item_count_pivot, min_support=support, use_colnames=True)

    metric ="lift"
    min_threshold = 1

    rules = association_rules(frequent_items, metric=metric, min_threshold=min_threshold)[['antecedents','consequents','support','confidence','lift']]
    rules.sort_values('confidence', ascending=False,inplace=True)

elif type(data) == type ("No result"):
    st.write("No Data")

def parse_list(x):
    x = list(x)
    if len(x) == 1 :
        return x[0]
    elif len(x) > 1:
        return ", ".join(x)
    
def return_item_df(item):
    #data = rules[["antecedents", "consequents"]].copy()
    #data["antecedents"] = data["antecedents"].apply(parse_list)
    #data["consequents"] = data["consequents"].apply(parse_list)

    #return list(data.loc[data["antecedents"]==item_antecedents].iloc[0:])
    datax = rules[["antecedents", "consequents"]].copy()
    datax["antecedents"] = datax["antecedents"].apply(parse_list)
    datax["consequents"] = datax["consequents"].apply(parse_list)
    cobax = datax.loc[datax['antecedents'] == item].iloc[:,1].tolist()
    hasil = ", ".join(cobax)
    return hasil



if type(data) != type ("No Result"):
    st.markdown("Hasil Rekomendasi: ")
    st.success(f"Jika Konsumen membeli **{item}**, maka konsumen tersebut membeli pula **{return_item_df(item)}**")
    