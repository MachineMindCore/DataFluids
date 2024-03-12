import pandas as pd
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import math

from datetime import datetime, date, timedelta

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from skforecast.ForecasterAutoreg import ForecasterAutoreg
from sklearn.metrics import mean_squared_error
from statsmodels.graphics.tsaplots import plot_acf
from sklearn.metrics import r2_score

from pydataxm import *
from datafluids.helper import local_css, download_link, df2st, data_expand, to_excel



TODAY = date.today()
PIVOT_DAY = datetime(TODAY.year - 1, 1, 1)

################################ Cuerpo #############################################


class DataFluidsApp:

    def __init__(self) -> None:
        st.set_page_config(layout='wide')
        st.sidebar.image('datafluids/style/logo_ver.png', use_column_width=True)     
        local_css("datafluids/style/style_gen.css") 
        
        self.client = pydataxm.ReadDB()
        self.tool = st.sidebar.selectbox('Herramientas Datafluids', ('Inicio', 'Predicciones'))

    def run(self) -> None:
        if self.tool == 'Inicio':
            self._home()
        
        if self.tool == 'Predicciones':
            self._prediction()

    def _home(self) -> None:
        local_css("datafluids/style/style_ini.css")
        st.image('datafluids/style/logo.png')
        col = st.columns(2)
        col[0].header('**¿Quienes somos?**')
        col[0].subheader('Datafluids es una compañía colombiana, que se dedica al desarrollo de software'
                        ' especializado en la industria eléctrica. Nuestras líneas principales de negocio'
                        ' son la predicción del comportamiento de la demanda de potencia del mercado eléctrico,'
                        ' además producimos aplicaciones que facilitan el análisis de los sistemas de potencia.')
        col[1].header('**Objetivo**')
        col[1].subheader('Nuestro objetivo es desarrollar herramientas computacionales que permitan a nuestros'
                        ' clientes mejorar la toma de decisiones estratégicas que permitan impactar el mercado.'
                        ' Además, facilitar la gestión operativa en el análisis de los sistemas de potencia.')
        col = st.columns(3)
        col[1].header('**Contactos**')
        col[1].subheader('**Sergio Andres Vargas Mendez**,')
        col[1].subheader('**Correo:** seavargasme@unal.edu.co')
        col[1].subheader('**Juan Jose Vargas Trujillo**,')
        col[1].subheader('**Correo:** jjvargast@unal.edu.co')
        col[1].subheader('**Carlos Andres Mendez Beltran**,')
        col[1].subheader('**Correo:** camendezb@unal.edu.co')
    
    def _prediction(self) -> None:
        
        PredSec = st.sidebar.radio('Prediccion: Secciones', ('Agentes', 'Explorador de datos',
                                                            'Predictor', 'Proy. Solar', 'Proy. Eolico'))

        if PredSec == 'Agentes':
            st.title('Agentes del mercado')
            AgDf = self.client.request_data(
                "DemaCome",           #Se indica el nombre de la métrica tal como se llama en el campo metricId
                "Agente",             #Se indica el nombre de la entidad tal como se llama en el campo Entity
                date(2022, 1, 1),  #Corresponde a la fecha inicial de la consulta
                TODAY  #Corresponde a la fecha final de la consulta
            )
            st.table(AgDf)

        if PredSec == 'Explorador de datos':
            st.title('Explorador')
            dini = st.date_input('Fecha inicial', value=date(2021, 1, 1), max_value=date(2021, 4, 30))
            dfin = st.date_input('Fecha final', value=date(2021, 2, 1), max_value=date(2021, 4, 30))
            # Peticion de datos
            MetaPowerDf = self.client.request_data('DemaCome', 1, dini, dfin)
            ORDf = MetaPowerDf.Values_code
            ORDf = ORDf.drop_duplicates()
            DefaultOR = int(np.where(ORDf == 'EMSC')[0])
            OR = st.selectbox('Codigo', ORDf, index=DefaultOR)  # Seleccion de Operador de Red

            if st.button('Visualizar'):
                PowerDf = MetaPowerDf[MetaPowerDf.Values_code == OR]
                PowerDf = PowerDf.drop(['Id', 'Values_code'], axis=1)

                BolsaDf = self.client.request_data('PrecBolsNaci', 0, dini, dfin)
                BolsaDf = BolsaDf.drop(['Id', 'Values_code'], axis=1)

                VolHydro = self.client.request_data('VoluUtilDiarEner', 0, dini, dfin)
                VolHydro = VolHydro.drop(columns=['Id'])

                [PowerTime, PowerData] = df2st(PowerDf)
                PowerData = PowerData / 1000
                [BolsaTime, BolsaData] = df2st(BolsaDf)
                [VolTime, VolData] = data_expand(VolHydro, 0, 1)
                VolData=VolData/1e9

                st.header('**Demanda de agente**')
                st.write(PowerDf)
                figPower = plt.figure(facecolor='#B3B4D1')
                plt.plot(PowerTime, PowerData)
                plt.grid('True')
                plt.ylabel('Demanda de agente (Mw)')
                plt.xlabel('Tiempo')
                plt.grid('True')
                figPower.autofmt_xdate(rotation=-90)
                st.pyplot(figPower)

                col = st.columns(2)

                col[0].header('**Precio nacional de bolsa**')
                col[0].write(BolsaDf)
                figBolsa = plt.figure(facecolor='#B3B4D1')
                plt.plot(BolsaTime, BolsaData)
                plt.grid('True')
                plt.ylabel('Precio nacional de bolsa ($COP/Kwh)')
                plt.xlabel('Tiempo')
                figBolsa.autofmt_xdate(rotation=-90)
                col[0].pyplot(figBolsa)

                col[1].header('**Volumen util diario**')
                col[1].write(VolHydro)
                figVol = plt.figure(facecolor='#B3B4D1')
                plt.plot(VolTime, VolData)
                plt.grid('True')
                plt.ylabel('Volumen util (Twh)')
                plt.xlabel('Tiempo')
                figVol.autofmt_xdate(rotation=90)
                col[1].pyplot(figVol)

                DownDf = pd.DataFrame({'Date': PowerTime, 'Power': PowerData, 'Bolsa': BolsaData, 'Volumen': VolData})
                download_link(DownDf)

        if PredSec == 'Predictor':

            # Opciones de prediccion
            st.title('Predictor')
            DatePm = st.date_input('Fecha de inicio (Prediccion)', value=datetime(2021, 1, 1),
                                min_value=datetime(2015, 1, 1),
                                max_value=datetime(2021, 4, 30))
            ORDf = pd.read_excel('AgA.xlsx')
            ORDf = ORDf['Código SIC']
            DefaultOR = int(np.where(ORDf == 'EMSC')[0])
            OR = st.selectbox('Seleccione agente de red', ORDf, index=DefaultOR)
            exo_var = st.selectbox('Seleccione una variable exogena para el modelo', ('Sin variable exogena',
                                                                                    'Precio de bolsa nacional',
                                                                                    'Volumen util diario'))
            TimeSel = st.radio('Seleccione tiempo de prediccion', ('Un dia', 'Una semana', 'Dos semanas', 'Un mes'))


            def switch(argument):
                switcher = {
                    'Un dia': 1,
                    'Una semana': 7,
                    'Dos semanas': 14,
                    'Un mes': 30
                }
                return switcher.get(argument)


            n_days = switch(TimeSel)
            PlusTime = timedelta(days=n_days)
            YearTime = timedelta(days=365)
            DatePf = DatePm + PlusTime
            DatePi = DatePm - YearTime

            if st.button('Calcular'):
                exVar = 0
                MetaPowerDf = self.client.request_data('DemaCome', 1, DatePi, DatePf)
                PowerDf = MetaPowerDf[MetaPowerDf.Values_code == OR]
                PowerDf = PowerDf.drop(['Id', 'Values_code'], axis=1)
                [PowerTime, PowerData] = df2st(PowerDf)
                PowerData = PowerData / 1000
                PowerSt = pd.DataFrame({'Time': PowerTime, 'Data': PowerData})

                if exo_var == 'Precio de bolsa nacional':
                    BolsaDf = self.client.request_data('PrecBolsNaci', 0, DatePi, DatePf)
                    BolsaDf = BolsaDf.drop(['Id', 'Values_code'], axis=1)
                    [BolsaTime, BolsaData] = df2st(BolsaDf)
                    BolsaSt = pd.DataFrame({'Time': BolsaTime, 'Data': BolsaData})
                    BolsaSt = BolsaSt.set_index('Time', inplace=False)
                    exo_x = BolsaSt
                    exVar = 1

                if exo_var == 'Volumen util diario':
                    VolHydro = self.client.request_data('VoluUtilDiarEner', 0, DatePi, DatePf)
                    VolHydro = VolHydro.drop(columns=['Id'])
                    [VolTime, VolData] = data_expand(VolHydro, 0, 1)
                    VolSt = pd.DataFrame({'Time': VolTime, 'Data': VolData})
                    VolSt = VolSt.set_index('Time', inplace=False)
                    exo_x = VolSt
                    exVar = 1

                x = PowerSt
                y = PowerSt['Data']
                x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=8784, shuffle=False)
                if exVar == 1:
                    exo_y = exo_x['Data']
                    exo_x_train, exo_x_test, exo_y_train, exo_y_test = train_test_split(exo_x, exo_y, train_size=8784,
                                                                                        shuffle=False)
                else:
                    exo_x_train, exo_x_test, exo_y_train, exo_y_test = [None, None, None, None]

                forecaster_rf = ForecasterAutoreg(
                    regressor=RandomForestRegressor(random_state=123),
                    lags=[1, 2, 3, 23, 24, 25, 47, 48, 49])

                forecaster_rf.fit(y=y_train, exog=exo_y_train)
                predicciones = forecaster_rf.predict_interval(steps=n_days*24, exog=exo_y_test, n_boot=20)
                data_pred = pd.DataFrame(data=predicciones, index=x_test.index)

                PredPlot = plt.figure(facecolor='#B3B4D1')
                plt.plot(PowerTime[8784:], y_test, label='Datos reales')
                plt.plot(PowerTime[8784:], data_pred.iloc[:,0], label='Prediccion')
                plt.fill_between(PowerTime[8784:],data_pred.iloc[:,1],data_pred.iloc[:,2],alpha=0.2,
                                color='#ff7700')
                plt.title('Prediccion de demanda')
                plt.ylabel('Demanda (MW)')
                plt.grid()
                PredPlot.autofmt_xdate(rotation=-90)
                PredPlot.legend()
                st.pyplot(PredPlot)

                col=st.columns(2)
                ACF1fig, ACFax = plt.subplots(facecolor='#B3B4D1')
                plot_acf(y_train, ax=ACFax, lags=60, use_vlines=True)
                plt.title('Autocorrelacion "train"')
                col[0].pyplot(ACF1fig)

                ACF2fig, ACFax = plt.subplots(facecolor='#B3B4D1')
                plot_acf(y_test, ax=ACFax, lags=60, use_vlines=True)
                plt.title('Autocorrelacion "test"')
                col[1].pyplot(ACF2fig)



                err_mse= (mean_squared_error(y_pred=data_pred.iloc[:,0], y_true=y_test)//0.01)/100
                err_r2=(r2_score(y_test, data_pred.iloc[:,0])//0.01)/100
                Comp = plt.figure(facecolor='#B3B4D1')
                plt.scatter(y_test, data_pred.iloc[:,0])
                plt.plot(y_test, y_test, color='#ff7700')
                plt.xlabel('Valores reales')
                plt.ylabel('Valores predichos')
                plt.title('Prediccion vs test '+'(MSE='+str(err_mse)+',  R2='+str(err_r2)+')')
                plt.grid()
                st.pyplot(Comp)
                DownDf = pd.DataFrame({'Date': PowerTime[8784:], 'Data_test': y_test, 'Data_pred': data_pred.iloc[:,0],
                                    'Lower limit':data_pred.iloc[:,1], 'Upper limit':data_pred.iloc[:,2]})
                download_link(DownDf)

        if PredSec == 'Proy. Solar':
            def PotOp(Irr, Vstc, Istc, Vnoct, Inoct):
                Irr_stc = 1000
                Irr_noct = 800
                Pstc = Vstc * Istc
                Pnoct = Vnoct * Inoct
                P = ((Irr - Irr_noct) / (Irr_stc - Irr_noct)) * (Pstc - Pnoct) + Pnoct
                if P < 0:
                    P = 0

                return P


            def Potday(Pan_frame, Irr_frame, Id, m, n_pan, Pplant):
                Irr_data = Irr_frame.iloc[:, m].to_numpy()
                PotData = np.zeros((3, 24))
                PanData = Pan_frame.iloc[[Id]]
                Vstc = PanData['Vmp STC'].to_numpy()
                Istc = PanData['Imp STC'].to_numpy()
                Vnoct = PanData['Vmp NOCT'].to_numpy()
                Inoct = PanData['Imp NOCT'].to_numpy()
                Pan_frame = Pan_frame.iloc[[Id]]
                r = Pan_frame[['r0', 'r10', 'r25']].to_numpy()
                r = r.squeeze()
                print(r)
                for i in range(0, 23):
                    PotData[0, i] = PotOp(Irr_data[i], Vstc, Istc, Vnoct, Inoct) * r[0] * n_pan / 1e8
                    PotData[1, i] = PotOp(Irr_data[i], Vstc, Istc, Vnoct, Inoct) * r[1] * n_pan / 1e8
                    PotData[2, i] = PotOp(Irr_data[i], Vstc, Istc, Vnoct, Inoct) * r[2] * n_pan / 1e8

                str_m = ['Enero', 'Febrero', 'Marzo', 'Abril', 'Mayo', 'Junio', 'Julio', 'Agosto', 'Septiembre',
                        'Octubre', 'Noviembre', 'Diciembre']
                h = range(24)
                Pot_r0 = PotData[0, :]
                Pot_r10 = PotData[1, :]
                Pot_r25 = PotData[2, :]
                Potfig = plt.figure(facecolor='#B3B4D1')
                ax = Potfig.add_subplot(1, 1, 1)
                plt.plot(h, Pot_r0, label='Rendimiento 0 años')
                plt.plot(h, Pot_r10, label='Rendimiento 10 años')
                plt.plot(h, Pot_r25, label='Rendimiento 25 años')
                plt.plot([0, 23], [Pplant, Pplant], 'k--', label='Potencia nominal')
                plt.title(str_m[m - 1])
                plt.xlabel('Tiempo (h)')
                plt.ylabel('Potencia (Mw)')
                plt.ylim([0, Pplant * 1.5])
                plt.legend()
                plt.grid()
                plt.plot()

                return [Potfig, Pot_r0, Pot_r10, Pot_r25, str_m[m - 1]]


            def MinPan(Irrframe, Pframe, Id, Pplant):
                minI = Irrframe.iloc[:, 1].sum()
                minMes = 1
                for m in range(1, 13):
                    if minI > Irrframe.iloc[:, m].sum() and Irrframe.iloc[:, m].sum() != 0:
                        minI = Irrframe.iloc[:, m].sum()
                        minMes = m
                maxI = Irrframe.iloc[0, minMes]
                for h in range(0, 24):
                    if maxI < Irrframe.iloc[h, minMes]:
                        maxI = Irrframe.iloc[h, minMes]
                PanData = Pframe.iloc[[Id]]
                Pmax = PotOp(maxI, PanData['Vmp STC'].tolist()[0], PanData['Imp STC'].tolist()[0],
                            PanData['Vmp NOCT'].tolist()[0], PanData['Imp NOCT'].tolist()[0])
                n_pan = Pplant / Pmax
                return n_pan


            def Dsummarize(Pframe, Id, Pplant, n_panel):
                PanData = Pframe.iloc[[Id]]
                col = st.columns(3)
                col[1].header('**Resumen**')
                col[1].header('**Producto: **' + str(PanData['Producto'].tolist()[0]))
                col[1].header('**Tecnologia: **' + str(PanData['Tecnologia'].tolist()[0]))
                col[1].header('**Potencia (pk): **' + str(PanData['Ppeak'].tolist()[0]) + ' Kwp')
                col[1].header('**Precio (und): **' + '$' + str(PanData['Precio'].tolist()[0]) + ' /und')
                col[1].header('**Potencia (planta): **' + str(Pplant / 1e6) + ' Mw')
                col[1].header('**Numero de paneles: **' + str(n_panel) + ' und')
                PPar = n_panel * PanData['Precio'].tolist()[0]
                col[1].header('**Precio parcial: **' + '$' + str(PPar // 1e6) + ' Mi')
                Ar=(PanData['D1'].tolist()[0])*(PanData['D2'].tolist()[0])/1e6
                Ar=((Ar*n_panel)//0.01)/100
                col[1].header('**Area minima (instalacion): **'+str(Ar)+' Km2')
                return


            st.title('Potencia de planta de solar')
            cities = pd.read_csv('Solar/cities.txt', header=None)
            CityId = st.selectbox('Seleccione ubicacion del proyecto', cities)
            IrrData = pd.read_excel('Solar/' + CityId + '.xlsx')
            FormatDf = pd.read_excel('Solar/Paneles.xlsx')
            st.markdown('Descargue aqui el formato: ')
            download_link(FormatDf)
            PanelFile = st.file_uploader("Subir especificaciones de paneles", type="xlsx")
            if PanelFile:
                PanelData = pd.read_excel(PanelFile)
                PanelId = st.selectbox('Seleccione el panel', PanelData['Producto'])
                PanelId = int(np.where(PanelData['Producto'] == PanelId)[0])
                Pplant = st.number_input('Potencia de planta (MW)', min_value=10, max_value=1000, value=500) * 1e6
                n_pan = math.ceil(MinPan(IrrData, PanelData, 1, Pplant))

            if st.button('Calcular'):
                DataDf = pd.DataFrame({'Hora': range(0, 24)})
                col = st.columns(2)
                for m in range(1, 7):
                    [Potfig, Pot_r0, Pot_r10, Pot_r25, m_name] = Potday(PanelData, IrrData, PanelId, m, n_pan, Pplant / 1e6)
                    DataDf.insert(loc=DataDf.shape[1], column=m_name + '_r0', value=Pot_r0)
                    DataDf.insert(loc=DataDf.shape[1], column=m_name + '_r10', value=Pot_r10)
                    DataDf.insert(loc=DataDf.shape[1], column=m_name + '_r25', value=Pot_r25)
                    col[0].pyplot(Potfig)
                for m in range(7, 13):
                    [Potfig, Pot_r0, Pot_r10, Pot_r25, m_name] = Potday(PanelData, IrrData, PanelId, m, n_pan, Pplant / 1e6)
                    DataDf.insert(loc=DataDf.shape[1], column=m_name + '_r0', value=Pot_r0)
                    DataDf.insert(loc=DataDf.shape[1], column=m_name + '_r10', value=Pot_r10)
                    DataDf.insert(loc=DataDf.shape[1], column=m_name + '_r25', value=Pot_r25)
                    col[1].pyplot(Potfig)

                Dsummarize(PanelData, PanelId, Pplant, n_pan)
                download_link(DataDf)

        if PredSec == 'Proy. Eolico':

            def MinAir(VData,pol,Pplant):
                Vmin=VData[0]
                for i in range(0,len(VData)):
                    if Vmin<VData[i] and VData[i]!=0:
                        Vmin=VData[i]
                Potmin=np.polyval(pol,Vmin)
                st.write(Potmin)
                n_gen=(Pplant)/(Potmin*1e3)
                n_gen=n_gen//1
                return n_gen

            def ZeroLim(v):
                for i in range(0,len(v)):
                    if v[i]<0:
                        v[i]=0
                return v


            st.title('Potencia de planta eolica')
            Vhor = pd.read_excel('Eolico/Vhoraria.xlsx').groupby('MUNICIPIO').mean()
            Vmen = pd.read_excel('Eolico/Vmensual.xlsx').groupby('MUNICIPIO').mean()
            cities=(Vhor.index).drop_duplicates()
            CityId = st.selectbox('Seleccione ubicacion del proyecto', cities)
            FormatDf = pd.read_excel('Eolico/generador.xlsx')
            st.markdown('Descargue aqui el formato: ')
            download_link(FormatDf)
            PanelFile = st.file_uploader("Subir especificaciones de aerogeneradores", type="xlsx")

            if PanelFile:
                AeroData = pd.read_excel(PanelFile)
                AeroId = st.selectbox('Seleccione el panel', AeroData['Producto'])
                AeroId = int(np.where(AeroData['Producto'] == AeroId)[0])
                Pplant = st.number_input('Potencia de planta (MW)', min_value=10, max_value=1000, value=500) * 1e6

            if st.button('Calcular'):
                AeroData=AeroData.iloc[AeroId,:]
                PData=(AeroData.loc['P0':'P15']).tolist()
                VData=(AeroData.loc['V0':'V15']).tolist()

                PVpol=np.polyfit(VData,PData,deg=len(VData)-1)
                PVcurve=np.polyval(PVpol,VData)

                PolFig=plt.figure(facecolor='#B3B4D1')
                plt.scatter(VData,PData)
                plt.plot(VData,PVcurve)
                plt.title('Curva Velocidad-Potencia')
                plt.ylabel('Potencia (Kw)')
                plt.xlabel('Velocidad (m/s)')
                st.pyplot(PolFig)

                Vmen = Vmen.loc[CityId,:].tolist()
                Vhor=Vhor.loc[CityId,:].tolist()
                n_gen=Pplant/(AeroData['Pnom']*1e3)
                n_gen=n_gen//1

                Phor=(np.polyval(PVpol,Vhor)*n_gen)/1000
                Phor=ZeroLim(Phor)
                Xhor=range(1,24)

                Pmen=(np.polyval(PVpol,Vmen)*n_gen)/1000
                Pmen=ZeroLim(Pmen)
                Xmen=range(1,13)

                col=st.columns(2)
                HorFig=plt.figure(facecolor='#B3B4D1')
                plt.plot(Xhor,Phor)
                plt.title('Potencia promedio horaria')
                plt.ylabel('Potencia (Mw)')
                plt.xlabel('Hora')
                col[0].pyplot(HorFig)

                MenFig=plt.figure(facecolor='#B3B4D1')
                plt.plot(Xmen,Pmen)
                plt.title('Potencia promedio mensual')
                plt.ylabel('Potencia (Mw)')
                plt.xlabel('Mes')
                col[1].pyplot(MenFig)

                HourDf=pd.DataFrame({'Vhor':Vhor,'Phor':Phor})
                MenDf=pd.DataFrame({'Vmen':Vmen,'Pmen':Pmen})

                st.markdown('Cantidad de Aerogeneradores: '+str(n_gen))
                st.markdown('Datos horarios:')
                download_link(HourDf)
                st.markdown('Datos Mensuales:')
                download_link(HourDf)






##########
# Datos de emisiones CO2 y CH4 dejaron de emitirse el 15 de mayo
# Razon sin justificar, no hay informacion
##########
# CO2Df=self.client.request_data("EmisionesCO2",0,dini,dt.date(2020,6,1))
# st.write(CO2Df)
# DateCol=CO2Df['Date']
# CO2Df=CO2Df.drop(columns=['Id','Values_Name','Values_code','Date'])
# CO2Df=CO2Df.apply(pd.to_numeric)
# CO2Df=CO2Df.set_index(DateCol,inplace=False)
# CO2Df=CO2Df.groupby(by='Date').sum()
# st.write(CO2Df)

# CH4Df=self.client.request_data('EmisionesCH4',0,dini,dfin)
# DateCol=CH4Df['Date']
# CH4Df=CH4Df.drop(columns=['Id','Values_Name','Values_code','Date'])
# CH4Df=CH4Df.apply(pd.to_numeric)
# CH4Df=CH4Df.set_index(DateCol,inplace=False)
# CH4Df=CH4Df.groupby(by='Date').sum()
# st.write(CH4Df)

