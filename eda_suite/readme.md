The Setup
You start with these lines in your main.ipynb:

Python

# 1. We load the dataframe
df = pd.read_csv('your_data.csv')

# 2. We create the container
container = EdaContainer()

# 3. We configure the container
container.config.df.from_value(df)
What has happened so far?

df = ...: A pandas DataFrame object is created and exists in memory. It is the only data object we have.

container = EdaContainer(): An EdaContainer object is created. Critically, at this point, NO DataProfiler, DataCleaner, or EDAAnalyzer objects exist. The container is just an object that holds a set of definitions or "recipes" (the providers) for how to build other objects later.

container.config...: You pass the df object into the container's configuration. The container updates its internal recipes. For example, the recipe for DataProfiler now knows that when it's asked to build a DataProfiler, it must use this specific df object as the dataframe argument.

After these three lines, you only have two main objects in memory: your df and the container. The stage is set, but the main event hasn't started.

The Main Event: Initialisation
This is the line that triggers everything:

Python

# 4. We ask the container to build the analyzer
analyzer = container.analyzer()
When do all the objects get initialised?
RIGHT NOW. This single line kicks off the following sequence:

1. Your code calls the analyzer() method on the container object.

2. The container looks at the recipe for analyzer. It sees the recipe is: providers.Factory(EDAAnalyzer, profiler=data_profiler, cleaner=data_cleaner, ...)

3. The container knows it cannot create an EDAAnalyzer until it has all the required ingredients (its dependencies): a profiler object, a cleaner object, a stats object, etc.

4. So, it goes down the ingredient list. First, it executes the recipe for data_profiler. It calls the DataProfiler class constructor (__init__) and passes the df to it. A DataProfiler object is now created (initialised).

5. Next, it executes the recipe for data_cleaner. It calls DataCleaner.__init__(dataframe=df). A DataCleaner object is now created.

6. It does this for StatisticsCalculator, Visualizer, and SchemaManager. All of these objects are now initialised and exist in memory.

7. Now that the container has successfully created all the necessary dependency objects, it can finally complete the original request. It calls EDAAnalyzer.__init__() and passes the objects it just created as arguments (profiler=..., cleaner=..., etc.).

8. The ```EDAAnalyzer``` object is now created.

9. This fully assembled analyzer object is returned and assigned to the analyzer variable in your notebook.

**The Roles of the Container and the Analyzer**
**What is the Container's role? **

The Container's role is to be the active builder and assembler.

- It knows the blueprints: It's the only place that knows a Visualizer needs a SchemaManager, or that a DataProfiler needs a DataFrame.

- It builds everything: It is responsible for calling the __init__ methods and creating all the objects.

- It wires everything together: Its most important job is connecting the objects correctly, passing the SchemaManager instance into the Visualizer, and passing all the specialist tools into the EDAAnalyzer.

You use it once at the start to build your fully-equipped tool.

**What is the Analyzer's role in this?**

The Analyzer's role is to be your simple User Interface (a Facade).

After the container has built everything, you are left with one powerful analyzer object. You don't need to juggle five different objects (profiler, cleaner, visualizer...) in your notebook.

- **It holds the tools:** The analyzer holds a reference to all the specialist objects the container gave it (self._profiler, self._cleaner, etc.).

- **It simplifies commands:** When you call a simple method like analyzer.show_profile(), you are giving an order to the analyzer.

- **It delegates the work:** The analyzer doesn't do the work itself. Its show_profile method turns around and calls self._profiler.get_summary(). It delegates the hard work to the correct specialist tool and then presents the result to you.

You use the analyzer for every task because it's your clean and simple control panel for the entire complex system the container built for you.