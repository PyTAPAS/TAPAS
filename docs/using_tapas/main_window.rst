Main Window
===========

TAPAS employs a tab-based interface in which each tab is dedicated to a distinct step of the analytical workflow.  Global project operations—such as saving/loading projects, adjusting GUI preferences, or selecting plotting templates—are accessible from the menu bar and toolbar above the tab pane.  In the backend these correspond to the `main_view` and `main_controller` modules.

- **Menu Bar / Toolbar Entries**  

  - ``Project → New Project`` / ``Open Project`` / ``Save Project``  / ``Save Project as`` 

    Save the current software- and data state to HDF including widget states, raw data, processed data, fitted data etc. Automatically adds metadata

  - ``Configurations → Open Config`` / ``Save Config``  / ``Save Config as`` 

    Config files hold the states of the widgets, eg. defalut values for buttons. Upon software start, the defalut configurations are loaded under ``\_internal\tapas\configurations\default_gui_config.json``. When a project is saved, the current widget states are also saved to the HDF file. 

  - ``Help → online Documentation`` 

    Link to the Read the Docs documentation.

  - ``Help → online Source Code`` 
  
    Link to the GitHub repository.

  - ``Help → local Source Code`` 
  
    Link to the  ``\_internal\tapas`` folder. The modules should be changed here. 


  - ``Help → Third-Party Acknowledgements`` / ``Third-Party Licenses`` / ``License``  / ``About`` 
  
    Information about the Licenses, Versions and Third-Party Packages, Graphics and Fonts. Points to the textfiles under ``\_internal\tapas``. Please do not change these when distributing the software. 


- **Status Bar & Error handling**  

  Status updates are displayed in green, expected error messages are displayed in red. These can be adjusted under ``\_internal\tapas\configurations\messages.py``. Unexpected or unhandled errors will be catched globally and the error message will be displayed in a pop-up window. In this case something went wrong and the current software state might be unstable. In this case, pleasse report the bug by opening a new issue at github. 

