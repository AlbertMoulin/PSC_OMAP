from statsmodels.tsa.arima_process import ArmaProcess

plt.subplot(2, 1, 1)
ar1 = np.array([1, -0.9038])
ma1 = np.array([1])
AR_object1 = ArmaProcess(ar1, ma1)
simulated_data_1 = AR_object1.generate_sample(nsample=1000)
plt.plot(simulated_data_1);    plt.show()
