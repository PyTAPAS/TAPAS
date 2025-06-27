Main Window
===========

TAPAS employs a tab-based interface in which each tab is dedicated to a distinct step of the analytical workflow.  Global project operations—such as saving/loading projects, adjusting GUI preferences, or selecting plotting templates—are accessible from the menu bar and toolbar above the tab pane.  In the backend these correspond to the `main_view` and `main_controller` modules.

- **Menu Bar / Toolbar Entries**  
  - ``Project → New Project`` / ``Open Project`` / ``Save Project``  / ``Save Project as`` 
  - ``Configurations → Open Config`` / ``Save Config``  / ``Save Config as`` 
      Config files hold the states of the widgets, eg. defalut values for buttons. Upon software start, the defalut configurations are loaded under ``\_internal\tapas\configurations\default_gui_config.json``
  - ``Help → Documentation`` / ``Source Code``
