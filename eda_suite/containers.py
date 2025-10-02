# ==============================================================================
# containers.py
#
# This module defines the Dependency Injection (DI) container for the EDA Suite.
# A DI container acts as a central "factory" or "assembly line" for an application.
# It knows how to create and wire together all the different classes (components)
# so that you don't have to do it manually in your main script. This practice,
# known as Inversion of Control (IoC), leads to more modular, flexible, and
# testable code.
#
# Author: Tim Vos
# Last Modified: 2 October 2025
# ==============================================================================

from dependency_injector import containers, providers
import pandas as pd

# Import all the classes that this container needs to know how to build.
from .profiler import DataProfiler
from .schema import SchemaManager
from .statistics import StatisticsCalculator
from .visualizer import Visualizer
from .analyzer import EDAAnalyzer
from .cleaner import DataCleaner
from .report_generator import ReportGenerator

class EdaContainer(containers.DeclarativeContainer):
    """
    The central DI container that manages the creation and lifecycle of all
    components in the EDA Suite.
    """

    # --- Configuration Provider ---
    # The Configuration provider acts as a placeholder for runtime values.
    # We use it here to create a 'slot' for the DataFrame, which will be provided
    # from the main notebook after the data has been loaded.
    config = providers.Configuration()

    # --- Component Providers ---
    # The following blocks are "providers". Each one is a blueprint that tells
    # the container how to create an instance of a specific class.

    # The Singleton provider ensures that only ONE instance of SchemaManager is
    # ever created per container. Any component that depends on it will receive
    # the exact same object. This is crucial for maintaining a consistent state,
    # such as our column name mappings, across the entire application.
    schema_manager = providers.Singleton(
        SchemaManager,
        # We configure it to receive its initial columns from the DataFrame
        # that will be supplied to our 'config' provider.
        initial_columns=config.df.provided.columns,
    )

    # The Factory provider creates a NEW instance of the class every time it is
    # requested. This is a good default for components that are mostly stateless
    # or where having a fresh instance for each operation is desirable.
    data_profiler = providers.Factory(
        DataProfiler,
        dataframe=config.df,
    )
    
    statistics_calculator = providers.Factory(
        StatisticsCalculator,
        dataframe=config.df,
    )

    visualizer = providers.Factory(
        Visualizer,
        dataframe=config.df,
        # This is dependency injection in action: we are telling the container
        # to provide the singleton instance of 'schema_manager' to the
        # Visualiser's constructor. This is the "wiring" part of the assembly.
        schema_manager=schema_manager,
    )

    data_cleaner = providers.Factory(
        DataCleaner,
        dataframe=config.df,
    )

    report_generator= providers.Factory(
        ReportGenerator,
    )

    # --- Final Product Provider ---
    # This is the provider for our main facade class. It is defined last because
    # it depends on all the other providers we've defined above. Python reads
    # this file from top to bottom, so all dependencies must exist before they
    # are referenced.
    analyzer = providers.Factory(
        EDAAnalyzer,
        profiler=data_profiler,
        stats=statistics_calculator,
        visualizer=visualizer, 
        schema=schema_manager,
        cleaner=data_cleaner,
        report_generator=report_generator,
    )