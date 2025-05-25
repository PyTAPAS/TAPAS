Software Architecture
=====================

We use a lightweight, MVC-inspired design: the View layer owns basic input handling and hosts our prebuilt tab-widget components, the Model is a minimal signal-emitter with just the stubs needed to notify the UI, and the Controller concentrates all of the complex logic—data analysis, fitting steps, workflow orchestration—so that anyone diving into the code need only look at the Controller to understand what the app really does. This loose separation gives you the benefits of MVC’s mental model without the ceremony of full, multi-class data models and view–model bindings


.. figure:: ../_static//TAPAS_architecture.svg
   :alt: Architecture overview
   :align: center
   :figwidth: 80%

   An overview of our MVC architecture.


.. code-block:: text

   _internal/
   ├── assets/
   │   ├── icons
   │   └── model previws
   ├── configurations/
   │   ├── gui input configurations
   │   ├── gui style configurations
   │   ├── plotting configurations
   │   └── logging configurations
   ├── controllers/
   │   ├── main controller
   │   ├── import controller
   │   ├── preprocessing controller
   │   └── ...
   ├── models/
   │   └── model
   ├── utils/
   │   ├── utils
   │   └── model functions
   ├── views/
   │   ├── tabs/
   |   |   ├──import tab
   |   |   ├──preprocessing tab
   |   |   └── ...
   │   ├── tabwidgets/
   |   |   ├──import tabwidgets
   |   |   ├──preprocessing tabwidgets
   |   |   └── ...
   │   └── main window
   └── ...
