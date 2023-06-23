import numpy as np
import matplotlib.pyplot as plt

# Çizgi parametreleri
gercek_egim = 2
gercek_kesen_nokta = 5

# Veri setini oluşturma
def veri_olustur(n, gercek_egim, gercek_kesen_nokta, hata_std):
    x = np.random.uniform(0, 10, n)
    hatalar = np.random.normal(0, hata_std, n)
    y = gercek_egim * x + gercek_kesen_nokta + hatalar
    return x, y

x, y = veri_olustur(100, gercek_egim, gercek_kesen_nokta, hata_std=2)

# Veriyi çizdirme
plt.scatter(x, y)
plt.xlabel('x')
plt.ylabel('y')
plt.show()


class CizgiModeli:
    def __init__(self):
        self.egim = np.random.randn()
        self.kesen_nokta = np.random.randn()

    def tahmin_et(self, x):
        return self.egim * x + self.kesen_nokta

    def gradyan(self, x, y):
        tahmin = self.tahmin_et(x)
        egim_gradyan = 2 * np.mean((tahmin - y) * x)
        kesen_nokta_gradyan = 2 * np.mean(tahmin - y)
        return egim_gradyan, kesen_nokta_gradyan

    def guncelle(self, egim_gradyan, kesen_nokta_gradyan, ogrenme_hizi):
        self.egim -= ogrenme_hizi * egim_gradyan
        self.kesen_nokta -= ogrenme_hizi * kesen_nokta_gradyan


# Eğitim fonksiyonu
def egit(model, x, y, ogrenme_hizi, epoch_sayisi):
    for epoch in range(epoch_sayisi):
        for i in range(len(x)):
            egim_gradyan, kesen_nokta_gradyan = model.gradyan(x[i], y[i])
            model.guncelle(egim_gradyan, kesen_nokta_gradyan, ogrenme_hizi)
        if epoch % 100 == 0:
            loss = np.mean((model.tahmin_et(x) - y) ** 2)
            print(f"Epoch {epoch}, Loss: {loss:.4f}")


# Modeli eğitme
model = CizgiModeli()
egit(model, x, y, ogrenme_hizi=0.001, epoch_sayisi=1001)

# Eğitilmiş modelin çizgisi
x_tahmin = np.linspace(0, 10, 100)
y_tahmin = model.tahmin_et(x_tahmin)

# Veriyi ve tahmini çizdirme
plt.scatter(x, y)
plt.plot(x_tahmin, y_tahmin, color='red')
plt.xlabel('x')
plt.ylabel('y')
plt.show()
