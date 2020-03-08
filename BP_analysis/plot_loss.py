import matplotlib.pyplot as plt

log_data = open('./c.txt')
content = log_data.readlines()

loss = []
i = 0
x =[]
for line in content:
    if '=' in line:
        loss_i = line.split('=')[1].strip('\n').strip()
        loss.append(float(loss_i))
        x.append(i)
        i = i+1

plt.figure(figsize=(10,8))
plt.plot(x,loss,c='r', lw=2,alpha=0.5)
plt.xlabel('Iterations(x10)')
plt.ylabel('Loss')
plt.show()