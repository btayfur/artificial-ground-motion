'''
Sabetta-Pugliese Modeli
Graizer-Kalkan 2015 (GK15) Modeli
Bozorgnia & Campbell, 2016 modeli
'''

import numpy as np
import math

class PGAEstimator:
    """
    PGA (Peak Ground Acceleration) tahmin fonksiyonlarını içeren sınıf.
    Farklı modeller kullanarak PGA değerlerini hesaplar.
    """
    
    @staticmethod
    def sabetta_pugliese(M: float, R: float, S: int) -> dict:
        """
        Calculate Peak Ground Acceleration (PGA) and Peak Ground Velocity (PGV) using Sabetta-Pugliese Model.

        Parameters:
            M (float): Magnitude of the earthquake.
            R (float): Distance to the seismic source (km).
            S (int): Soil type (0 for rock, 1 for alluvium).

        Returns:
            dict: A dictionary containing calculated PGA (m/s^2) and PGV (cm/s).
        """
        # Calculate Peak Ground Acceleration (PGA)
        log_A = 1.344 + 0.328 * M - 1.09 * math.log10(math.sqrt(R**2 + 5**2)) + 0.262 * S
        PGA = 10 ** log_A


        return round(PGA/98.1, 4)
    
    @staticmethod
    def graizer_kalkan_2015(magnitude, distance, depth=10.0, vs30=760.0):
        """
        Graizer-Kalkan 2015 (GK15) modelini kullanarak PGA hesaplama.
        
        Parametreler:
        -------------
        magnitude : float
            Depremin büyüklüğü (Mw)
        distance : float
            Rjb uzaklığı (km) - Joyner-Boore mesafesi
        depth : float, optional
            Depremin odak derinliği (km)
        vs30 : float, optional
            30 metre derinliğe kadar ortalama kayma dalgası hızı (m/s)
            
        Referans:
        ---------
        Graizer, V., & Kalkan, E. (2015). Update of the Graizer–Kalkan ground-motion 
        prediction equations for shallow crustal continental earthquakes.
        
        Dönüş:
        ------
        float: PGA değeri (g biriminde)
        """
        # Burada gerçek GK15 modeli formülü uygulanmalıdır
        # Aşağıdaki implementasyon basitleştirilmiştir
        
        # Temel parametreler
        c1 = -3.25 + 0.8 * magnitude
        c2 = 0.9
        
        # Uzaklık ve derinlik etkisi
        r_eff = np.sqrt(distance**2 + depth**2)
        attenuation = c2 * np.log10(r_eff/10.0)
        
        # Zemin etkisi
        site_term = 0.0
        if vs30 < 760.0:
            site_term = 0.35 * np.log10(760.0/vs30)
        
        # PGA hesaplama (g biriminde)
        log_pga = c1 - attenuation + site_term
        pga = 10 ** log_pga
        
        return pga
    
    @staticmethod
    def campbell_bozorgnia_2014(magnitude, distance, vs30=760.0, depth=10.0, fault_type='SS'):
        """
        Campbell & Bozorgnia (2014/2016) modelini kullanarak PGA hesaplama.
        
        Parametreler:
        -------------
        magnitude : float
            Depremin büyüklüğü (Mw)
        distance : float
            Rrup uzaklığı (km) - Yırtılma yüzeyine en kısa uzaklık
        vs30 : float, optional
            30 metre derinliğe kadar ortalama kayma dalgası hızı (m/s)
        depth : float, optional
            Depremin odak derinliği (km)
        fault_type : str, optional
            Fay tipi: 'SS' (Strike-Slip), 'N' (Normal), 'R' (Reverse)
            
        Referans:
        ---------
        Campbell, K. W., & Bozorgnia, Y. (2014). NGA-West2 ground motion model for the 
        average horizontal components of PGA, PGV, and 5% damped linear acceleration 
        response spectra. Earthquake Spectra, 30(3), 1087-1115.
        
        Dönüş:
        ------
        float: PGA değeri (g biriminde)
        """
        # Model parametreleri
        c0 = -4.416
        c1 = 0.984
        c2 = 0.537
        c3 = -1.499
        c4 = -0.496
        c5 = -2.773
        c6 = 0.248
        c7 = 6.768
        
        # Fay tipi katsayıları
        f_fault = 0.0
        if fault_type == 'R':
            f_fault = 0.251
        elif fault_type == 'N':
            f_fault = -0.053
        
        # Zemin etkisi
        f_site = 0.0
        if vs30 < 1500.0:
            f_site = c5 * np.log(vs30/1500.0)
        
        # Büyüklük etkisi
        if magnitude <= 5.5:
            f_mag = c0 + c1 * magnitude
        elif magnitude <= 6.5:
            f_mag = c0 + c1 * magnitude + c2 * (magnitude - 5.5)
        else:
            f_mag = c0 + c1 * magnitude + c2 * (magnitude - 5.5) + c3 * (magnitude - 6.5)
        
        # Uzaklık etkisi
        r_term = np.sqrt(distance**2 + c7**2)
        f_dis = c4 * np.log(r_term) + (c6 * depth)
        
        # PGA hesaplama (ln(g) biriminde)
        ln_pga = f_mag + f_dis + f_site + f_fault
        
        # PGA (g biriminde)
        pga = np.exp(ln_pga)
        
        return pga
    
    def estimate_pga_for_tdby(Ss, S1, soil_class, c1=2.5, c2=0.5, tune=False):
        '''
        This function estimates PGA for TDBY using empirical formula.
        It is just an empirical formula, it will be tuned after optimization.
        
        Parameters:
        Ss: float
            Stiff soil factor
        S1: float
            Soft soil factor
        soil_class: str
            Soil class
        c1: float
            Stiff soil factor. if tune is not true, it will be ignored.
        c2: float
            Soft soil factor. if tune is not true, it will be ignored.
        tune: bool
            Let it false. Dont use it. Its just for optimization phase.
        '''
        if not tune:
            if soil_class == 'ZA':
                c1 = 3.0  # Higher ratio for stiff soils
                c2 = 0.4  # Less effective for soft soils
            elif soil_class == 'ZB':
                c1 = 2.8
                c2 = 0.45
            elif soil_class == 'ZC':
                c1 = 2.5
                c2 = 0.5
            elif soil_class == 'ZD':
                c1 = 2.0  
                c2 = 0.6  
            elif soil_class == 'ZE':
                c1 = 1.8  
                c2 = 0.7  

        # Its just an empirical formula, it will be tuned after optimization.
        PGA = (Ss / c1 + S1 / c2) / 2
        return PGA




