# pip install --index-url=https://blpapi.bloomberg.com/repository/releases/python/simple/ blpapi
import blpapi
import pandas as pd
import numpy as np
import datetime as dt
from datetime import datetime
from dateutil.relativedelta import relativedelta

DATE = blpapi.Name("date")
ERROR_INFO = blpapi.Name("errorInfo")
EVENT_TIME = blpapi.Name("EVENT_TIME")
FIELD_DATA = blpapi.Name("fieldData")
FIELD_EXCEPTIONS = blpapi.Name("fieldExceptions")
FIELD_ID = blpapi.Name("fieldId")
SECURITY = blpapi.Name("security")
SECURITY_DATA = blpapi.Name("securityData")

class BLP():
    #-----------------------------------------------------------------------------------------------------    
   
    def __init__(self):
        """
            Improve this
            BLP object initialization
            Synchronus event handling
           
        """
        # Create Session object
        self.session = blpapi.Session()
       
       
        # Exit if can't start the Session
        if not self.session.start():
            print("Failed to start session.")
            return
       
        # Open & Get RefData Service or exit if impossible
        if not self.session.openService("//blp/refdata"):
            print("Failed to open //blp/refdata")
            return
       
        self.session.openService('//BLP/refdata')
        self.refDataSvc = self.session.getService('//BLP/refdata')
 
        #print('Session open')
   
    #-----------------------------------------------------------------------------------------------------
   
    def bdh(self, strSecurity, strFields, startdate, enddate, per='DAILY', perAdj = 'CALENDAR', days = 'NON_TRADING_WEEKDAYS', fill = 'PREVIOUS_VALUE', currency = ""):
        """
            Summary:
                HistoricalDataRequest ;
       
                Gets historical data for a set of securities and fields
 
            Inputs:
                strSecurity: list of str : list of tickers
                strFields: list of str : list of fields, must be static fields (e.g. px_last instead of last_price)
                startdate: date
                enddate
                per: periodicitySelection; daily, monthly, quarterly, semiannually or annually
                perAdj: periodicityAdjustment: ACTUAL, CALENDAR, FISCAL
                curr: string, else default currency is used
                Days: nonTradingDayFillOption : NON_TRADING_WEEKDAYS*, ALL_CALENDAR_DAYS or ACTIVE_DAYS_ONLY
                fill: nonTradingDayFillMethod :  PREVIOUS_VALUE, NIL_VALUE
               
                Options can be selected these are outlined in “Reference Services and Schemas Guide.”    
           
            Output:
                A list containing as many dataframes as requested fields
            # Partial response : 6
            # Response : 5
           
        """
           
        #-----------------------------------------------------------------------
        # Create request
        #-----------------------------------------------------------------------
       
        # Create request
        request = self.refDataSvc.createRequest('HistoricalDataRequest')
       
        # Put field and securities in list is single value is passed
        if type(strFields) == str:
            strFields = [strFields]
           
        if type(strSecurity) == str:
            strSecurity = [strSecurity]
   
        # Append list of securities
        for strF in strFields:
            request.append('fields', strF)
   
        for strS in strSecurity:
            request.append('securities', strS)
   
        # Set other parameters
        request.set('startDate', startdate.strftime('%Y%m%d'))
        request.set('endDate', enddate.strftime('%Y%m%d'))
        request.set('periodicitySelection', per)
        request.set('periodicityAdjustment', perAdj)
        request.set('nonTradingDayFillMethod', fill)
        request.set('nonTradingDayFillOption', days)
        if(currency!=""):  
            request.set('currency', currency)
 
        #-----------------------------------------------------------------------
        # Send request
        #-----------------------------------------------------------------------
 
        requestID = self.session.sendRequest(request)
        print("Sending request")
       
        #-----------------------------------------------------------------------
        # Receive request
        #-----------------------------------------------------------------------
       
        dict_Security_Fields={}
        liste_msg = []
        while True:
            event = self.session.nextEvent()
           
            # Ignores anything that's not partial or final
            if (event.eventType() !=blpapi.event.Event.RESPONSE) & (event.eventType() !=blpapi.event.Event.PARTIAL_RESPONSE):
                continue
           
            # Extract the response message
            msg = blpapi.event.MessageIterator(event).__next__()
            liste_msg.append(msg)
            # Break loop if response is final
            if event.eventType() == blpapi.event.Event.RESPONSE:
                break
   
        #-----------------------------------------------------------------------
        # Exploit data
        #----------------------------------------------------------------------
       
        # Create dictionnary per field
        dict_output = {}
        for field in strFields:
            dict_output[field] = {}
            for ticker in strSecurity:
                dict_output[field][ticker] = {}
                 
        # Loop on all messages
        for msg in liste_msg:
            countElement = 0
            security_data = msg.getElement(SECURITY_DATA)
            security = security_data.getElement(SECURITY).getValue() #Ticker
            # Loop on dates
            for field_data in security_data.getElement(FIELD_DATA):
               
                # Loop on differents fields
                date = field_data.getElement(0).getValue()
               
                for i in range(1,field_data.numElements()):
                    field = field_data.getElement(i)
                    dict_output[str(field.name())][security][date] = field.getValue()
                   
                countElement = countElement+1 if field_data.numElements()>1 else countElement
                
            # remove ticker
            if countElement==0:
                for field in strFields:
                    del dict_output[field][security]
                   
        for field in dict_output:
            dict_output[field] = pd.DataFrame.from_dict(dict_output[field])
        return dict_output  

    #-----------------------------------------------------------------------------------------------------

    def bdp(self, strSecurity, strFields, strOverrideField='', strOverrideValue=''):
        
        """
            Summary:
                Reference Data Request ; Real-time if entitled, else delayed values 
                Only supports 1 override
                
                
            Input:
                strSecurity
                strFields
                strOverrideField
                strOverrideValue         
            
            Output:
               Dict 
        """
        
        #-----------------------------------------------------------------------
        # Create request
        #-----------------------------------------------------------------------
        
        # Create request
        request = self.refDataSvc.createRequest('ReferenceDataRequest')
        
        # Put field and securities in list is single field passed
        if type(strFields) == str:
            strFields = [strFields]
        
        if type(strSecurity) == str:
            strSecurity = [strSecurity]
            
        # Append list of fields
        for strD in strFields:
            request.append('fields', strD)

        # Append list of securities
        for strS in strSecurity:
            request.append('securities', strS)

        # Add override 
        if strOverrideField != '':
            o = request.getElement('overrides').appendElement()
            o.setElement('fieldId', strOverrideField)
            o.setElement('value', strOverrideValue)

        #-----------------------------------------------------------------------
        # Send request
        #-----------------------------------------------------------------------

        requestID = self.session.sendRequest(request)
        # print("Sending request")

        #-----------------------------------------------------------------------
        # Receive request                
        #-----------------------------------------------------------------------
                
        list_msg = []
        
        while True:
            event = self.session.nextEvent()
            
            # Ignores anything that's not partial or final
            if (event.eventType() !=blpapi.event.Event.RESPONSE) & (event.eventType() !=blpapi.event.Event.PARTIAL_RESPONSE):
                continue
            
            # Extract the response message
            msg = blpapi.event.MessageIterator(event).__next__()
            list_msg.append(msg)
            # Break loop if response is final
            if event.eventType() == blpapi.event.Event.RESPONSE:
                break    

        #-----------------------------------------------------------------------
        # Extract the data 
        #-----------------------------------------------------------------------
        dict_output = {}        
        for msg in list_msg:

            for security_data in msg.getElement(SECURITY_DATA):
                ticker = security_data.getElement(SECURITY).getValue() #Ticker
                dict_output[ticker] = {}
                for i in range(0, security_data.getElement(FIELD_DATA).numElements()): # on boucle sur les fields
                    fieldData = security_data.getElement(FIELD_DATA).getElement(i)
                    dict_output[ticker][str(fieldData.name())] = fieldData.getValue()


        return pd.DataFrame.from_dict(dict_output).T
    
    def bds(self, strSecurity, strFields, strOverrideField='', strOverrideValue=''):
        
        """
            Summary:
                Reference Data Request ; Real-time if entitled, else delayed values 
                Only supports 1 override
                
                
            Input:
                strSecurity
                strFields
                strOverrideField
                strOverrideValue         
            
            Output:
               Dict 
        """
        
        #-----------------------------------------------------------------------
        # Create request
        #-----------------------------------------------------------------------
        
        # Create request
        request = self.refDataSvc.createRequest('ReferenceDataRequest')
        
        # Put field and securities in list is single field passed
        if type(strFields) == str:
            strFields = [strFields]
        
        if type(strSecurity) == str:
            strSecurity = [strSecurity]
            
        # Append list of fields
        for strD in strFields:
            request.append('fields', strD)

        # Append list of securities
        for strS in strSecurity:
            request.append('securities', strS)

        # Add override 
        if strOverrideField != '':
            o = request.getElement('overrides').appendElement()
            o.setElement('fieldId', strOverrideField)
            o.setElement('value', strOverrideValue)

        #-----------------------------------------------------------------------
        # Send request
        #-----------------------------------------------------------------------

        requestID = self.session.sendRequest(request)
        #print("Sending request")

        #-----------------------------------------------------------------------
        # Receive request                
        #-----------------------------------------------------------------------
                
        list_msg = []
        
        while True:
            event = self.session.nextEvent()
            
            # Ignores anything that's not partial or final
            if (event.eventType() !=blpapi.event.Event.RESPONSE) & (event.eventType() !=blpapi.event.Event.PARTIAL_RESPONSE):
                continue
            
            # Extract the response message
            msg = blpapi.event.MessageIterator(event).__next__()
            list_msg.append(msg)
            # Break loop if response is final
            if event.eventType() == blpapi.event.Event.RESPONSE:
                break    
        
        #-----------------------------------------------------------------------
        # Extract the data 
        #-----------------------------------------------------------------------
        dict_output = {}
        for msg in list_msg:
            for security_data in msg.getElement(SECURITY_DATA): #Boucle sur les tickers
                ticker = security_data.getElement(SECURITY).getValue() #Ticker
                dict_field = []
                for field_data in security_data.getElement(FIELD_DATA).getElement(0): #Boucle sur les données des fields
                    
                    dict_elements = {}
                    for i in range(field_data.numElements()): 
                        dict_elements[str(field_data.getElement(i).name())] = str(field_data.getElement(i).getValue())
                    dict_field.append(dict_elements)
                
                dict_output[ticker] = pd.DataFrame(dict_field)
        
        return dict_output
    
    #-----------------------------------------------------------------------------------------------------
 
    def closeSession(self):    
        #print("Session closed")
        self.session.stop()

class DataLoader():
    """
    DataLoader class to handle data loading from Bloomberg using the BLP class.
    """

    def load_tickers_index(self, index : str, start_date_dt : datetime =None, end_date_dt : datetime =None):
        """
        Charge les données historiques des membres d'un indice à chaque fin de mois 
        entre start_date_dt et end_date_dt. Retourne un DataFrame avec les tickers 
        en colonnes et 1/0 indiquant leur présence à chaque date.
        """
        # Génère les dates de fin de mois entre les deux bornes
        dates = pd.date_range(start = start_date_dt, end = end_date_dt, freq='M')
        dates_str = dates.strftime('%Y%m%d')  # Format 'YYYYMMDD'
        
        # Assure que l'index est une liste
        if isinstance(index, str):
            index = [index]
        
        field = "INDX_MWEIGHT_HIST"
        df_fin = pd.DataFrame(index=dates_str)
        blp = BLP()

        for date_str in dates_str:
            # Récupère les données Bloomberg avec override sur la date
            df = blp.bds(
                strSecurity=index,
                strFields=[field],
                strOverrideField="END_DATE_OVERRIDE",
                strOverrideValue=date_str
            )
            
            # Récupère les tickers uniques pour la date et les ajoute si nécessaire
            tickers = df[index[0]]['Index Member'].unique()
            for ticker in tickers:
                if ticker not in df_fin.columns:
                    df_fin[ticker] = 0
            
            # Marque la présence des tickers à cette date
            df_fin.loc[date_str, tickers] = 1
        blp.closeSession()
        return df_fin


    def load_historical_data(self, tickers, fields, start_date=None, end_date=None):
        """
        Récupère les données Bloomberg pour une liste de tickers et de champs.
        Décale de 3 mois tous les champs sauf 'PX_LAST' dans le temps (shift temporel).
        """
        blp = BLP()

        # Récupération des données
        df = blp.bdh(
            strSecurity=tickers,
            strFields=fields,
            startdate=start_date,
            enddate=end_date,
            per='DAILY'
        )

        # Vérifie que l'index est bien temporel
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)

        # Traitement des données si multi-index (cas multi-ticker avec champs en colonnes hiérarchiques)
        if isinstance(df.columns, pd.MultiIndex):
            for field in fields:
                if field != 'PX_LAST' and field in df.columns.levels[0]:
                    for ticker in df[field].columns:
                        df[(field, ticker)] = df[(field, ticker)].shift(periods=3, freq='M')
        else:
            # Cas simple : colonnes plates
            for field in fields:
                if field != 'PX_LAST' and field in df.columns:
                    df[field] = df[field].shift(periods=3, freq='M')

        blp.closeSession()
        return df

