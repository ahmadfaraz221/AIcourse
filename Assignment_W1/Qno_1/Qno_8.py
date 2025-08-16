#calculat the loss and proft

cp = 150
sp = 120
if sp > cp:

    profit = sp - cp
    print("profit of:",profit)
elif cp > sp:
    loss = cp - sp 
    print("loss of:",loss)
else:
    print("no profit;no loss.")