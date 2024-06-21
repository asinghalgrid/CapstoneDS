from langchain_groq import ChatGroq
from langchain.agents import initialize_agent

from io import StringIO, BytesIO

import matplotlib.pyplot as plt

from langchain.pydantic_v1 import BaseModel, Field
from langchain.tools import BaseTool
from typing import Any, Optional, Type

from langchain.callbacks.manager import CallbackManagerForToolRun

import pandas as pd
import numpy as np

from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA

llm_model = ChatGroq(temperature=0, groq_api_key='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', model_name="mixtral-8x7b-32768")

PREFIX = """Your task is to perform time series forecasting using ARIMA. 
The input will be the data in JSON format but only take the values in the column specified as a list.
First ensure that the input is stationary and if not then convert it into stationary data.
You have access to the following tools:

Use the observations from the tools used to determine the next steps.
After using the ARIMA tool, invert the differencing process if the data was converted into stationary data.
Then the final step is to take the observation from the previous tool and plot a line chart and return a CSV file.

At any step if an action cannot be executed then simply state the error clearly"""

FORMAT_INSTRUCTIONS = """Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: Task finished"""

SUFFIX = """Begin!

Question: {input}
Thought:{agent_scratchpad}"""


def invert_differencing(diff_series, original_first_value):
    cum_sum = np.cumsum(diff_series)

    # Add the original first value to the cumulative sum to reconstruct the original series
    inverted_series = cum_sum + original_first_value

    # Prepend the original first value to the beginning of the series
    inverted_series = np.insert(inverted_series, 0, original_first_value)

    return pd.Series(inverted_series).to_json()

#Creating a custom tool to check for stationarity in the data
class InputData(BaseModel):
    input_data: str = Field(description='The list of values in JSON format to take in as input')

class StationaryCheck(BaseTool):
    name = "StationaryCheck"
    description = "Useful for when you need to check the stationarity of the data"
    args_schema: Type[BaseModel] = InputData

    def _run(self, input_data: str, run_manager: Optional[CallbackManagerForToolRun] = None) -> str:
        # data_dict = json.loads(input_data)
        # df = pd.DataFrame.from_dict(data_dict)
        series = pd.read_json(StringIO(input_data),typ='series')
        # ts = df.iloc[:,1].values
        dftest = adfuller(series.values)
        adf = dftest[0]
        pvalue = dftest[1]
        critical_value = dftest[4]['5%']
        if (pvalue < 0.05) and (adf < critical_value):
            return True
        else:
            return False

class StationaryConverter(BaseTool):
    name = "StationaryConverter"
    description = "Useful for converting non stationary data into stationary data"
    args_schema: Type[BaseModel] = InputData

    def _run(self, input_data: list, run_manager: Optional[CallbackManagerForToolRun] = None) -> str:
        # data_dict = json.loads(input_data)
        # df = pd.DataFrame.from_dict(data_dict)
        series = pd.read_json(StringIO(input_data),typ='series')
        # ts = df.iloc[:,1]

        return pd.Series(np.diff(series.values)).to_json()


class ARIMATool(BaseTool):
    name = "ARIMATool"
    description = "Time series forecasting using the ARIMA model"
    args_schema: Type[BaseModel] = InputData

    def _run(self, input_data: str, converted: bool = True,
             run_manager: Optional[CallbackManagerForToolRun] = None) -> list:
        # data_dict = json.loads(input_data)
        # df = pd.DataFrame.from_dict(data_dict)
        # data = df.iloc[:,1].values
        data = pd.read_json(StringIO(input_data), typ='series').values
        index = int(0.7 * len(data))
        train = data[:index]
        test = data[index:]
        history = [x for x in train]
        predictions = []

        for t in range(len(test)):
            model = ARIMA(history, order=(3, 1, 0))
            model.initialize_approximate_diffuse()
            model_fit = model.fit()
            output = model_fit.forecast()
            yhat = output[0]
            predictions.append(yhat)
            observation = test[t]
            history.append(observation)

        for i in range(3):
            model = ARIMA(history, order=(3, 1, 0))
            model.initialize_approximate_diffuse()
            model_fit = model.fit()
            output = model_fit.forecast()
            yhat = output[0]
            predictions.append(yhat)
            obs = test[t]
            history.append(obs)

        return pd.Series(list(data) + predictions[-3:]).to_json()

class InvertDifferencing(BaseTool):
    name = "InvertDifferencing"
    description = "If the data was converted into stationary then we invert the differencing process"
    args_schema: Type[BaseModel] = InputData

    def _run(self, input_data: str, run_manager: Optional[CallbackManagerForToolRun] = None) -> Any:
        return invert_differencing(pd.read_json(StringIO(input_data),typ='series').values,original_value)


class PresentData(BaseTool):
    name = "PresentData"
    description = "Plot a line chart and return a CSV file"
    args_schema: Type[BaseModel] = InputData

    def _run(self, input_data: str, run_manager: Optional[CallbackManagerForToolRun] = None) -> None:
        data = pd.read_json(StringIO(input_data), typ='series').values

        indices = list(range(len(data)))

        buf = BytesIO()
        plt.plot(indices[:-3], data[:-3], color='blue', label='History')
        plt.plot(indices[-4:], data[-4:], color='deeppink', label='Predicted')
        plt.xlabel('Index')
        plt.ylabel('Value')
        plt.legend()
        plt.savefig(buf, format='png')
        # plt.show()

        pd.Series(data).to_csv('time_series.csv')


tools = [StationaryCheck(), StationaryConverter(), ARIMATool(), InvertDifferencing(), PresentData()]

# Create the agent

agent = initialize_agent(tools,llm_model,agent="zero-shot-react-description",verbose=True,
        agent_kwargs={
        'prefix':PREFIX,
        'format_instructions':FORMAT_INSTRUCTIONS,
        'suffix':SUFFIX})

def invoke_agent(path, column):
    df = pd.read_csv(path)
    input_data = df[:90].to_json()
    global original_value
    original_value = df[column][0]
    agent.invoke(f'This is the data:\n{input_data}.\nPerform the task on the {column} column')