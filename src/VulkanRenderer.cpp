#include "VulkanRenderer.hpp"
#include <cstring>
#include <iomanip>
#include <sstream>

// Defines the directory where compiled shader files (.spv) are located.
#ifndef SHADER_DIR
#define SHADER_DIR "shaders/"
#endif

// Validation layers are optional components that hook into Vulkan function calls
// to apply additional operations, like checking for errors/misuse of the API.
const std::vector<const char*> validationLayers = {
    "VK_LAYER_KHRONOS_validation"
};

// List of device extensions we need. SWAPCHAIN is essential to present images to the screen.
const std::vector<const char*> deviceExtensions = {
    VK_KHR_SWAPCHAIN_EXTENSION_NAME
};

// Paths to our input raw data sequences.
const std::string COLOR_PATH_PREFIX  = "nvt_2026_01_23_11_43_31_45/color_input_0_";
const std::string DEPTH_PATH_PREFIX  = "nvt_2026_01_23_11_43_31_45/depth_input_0_";
const std::string NORMAL_PATH_PREFIX = "nvt_2026_01_23_11_43_31_45/normal_input_0_";
const std::string ALBEDO_PATH_PREFIX = "nvt_2026_01_23_11_43_31_45/albedo_0_"; // New Path
const std::string MV_PATH_PREFIX     = "nvt_2026_01_23_11_43_31_45/mv_input_0_";
const std::string FILE_EXTENSION = ".raw";
const uint32_t FILE_STRIDE = 4; // 4 bytes per pixel (RGBA)

// Enable validation layers only in Debug builds to save performance in Release.
#ifdef NDEBUG
const bool enableValidationLayers = false;
#else
const bool enableValidationLayers = true;
#endif

// Helper proxy function to create the Debug Utils Messenger extension object.
// Extensions sometimes need to be looked up by name because they aren't part of the core Vulkan pointer table.
VkResult CreateDebugUtilsMessengerEXT(VkInstance instance, const VkDebugUtilsMessengerCreateInfoEXT* pCreateInfo, const VkAllocationCallbacks* pAllocator, VkDebugUtilsMessengerEXT* pDebugMessenger) {
    auto func = (PFN_vkCreateDebugUtilsMessengerEXT) vkGetInstanceProcAddr(instance, "vkCreateDebugUtilsMessengerEXT");
    if (func != nullptr) {
        return func(instance, pCreateInfo, pAllocator, pDebugMessenger);
    } else {
        return VK_ERROR_EXTENSION_NOT_PRESENT;
    }
}

// Helper proxy function to destroy the Debug Utils Messenger.
void DestroyDebugUtilsMessengerEXT(VkInstance instance, VkDebugUtilsMessengerEXT debugMessenger, const VkAllocationCallbacks* pAllocator) {
    auto func = (PFN_vkDestroyDebugUtilsMessengerEXT) vkGetInstanceProcAddr(instance, "vkDestroyDebugUtilsMessengerEXT");
    if (func != nullptr) {
        func(instance, debugMessenger, pAllocator);
    }
}

// Main class entry point.
// Orchestrates the application lifecycle: Init Window -> Init Vulkan -> Run Loop -> Cleanup.
void VulkanRenderer::run() {
    initWindow();
    initVulkan();
    mainLoop();
    cleanup();
}

// Initialize the GLFW library and create a window.
// We tell GLFW *not* to create an OpenGL context because we are using Vulkan.
void VulkanRenderer::initWindow() {
    glfwInit();
    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API); // No OpenGL
    glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE);   // Disable resizing for simplicity
    window = glfwCreateWindow(WIDTH, HEIGHT, "Vulkan Image Sequence Player", nullptr, nullptr);
}

// Master initialization function. Calls all the sub-init functions in the required order.
// Vulkan is very explicit; everything needs to be created manually.
void VulkanRenderer::initVulkan() {
    createInstance();           // The connection between our app and the Vulkan library.
    setupDebugMessenger();      // Setup error logging.
    createSurface();            // The interface between Vulkan and the Window System.
    pickPhysicalDevice();       // Select a graphics card (GPU).
    createLogicalDevice();      // Create a logical interface to the selected GPU.
    createSwapchain();          // Create the chain of images that will be presented to the screen.
    createImageViews();         // Create views (wrappers) for the swapchain images so the pipeline can see them.
    createRenderPass();         // Define the structure of a rendering pass (attachments, subpasses, dependencies).
    createDescriptorSetLayout();// Define the "signatures" of shaders (what resources they expect).
    createFinalDescriptorSetLayout();
    createCommandPool();        // Pool memory for allocating commands.
    
    // Create resources for offscreen passes (Ray Marching, Denoising, etc.)
    createOffscreenResources(); 
    createDepthDSResources();
    
    createGraphicsPipeline();   // Create the pipeline state objects (shaders, blending, rasterization settings).
    createFramebuffers();       // Connect image views to the render pass attachments.
    
    // Create texture resources (Images, Views, Samplers) on the GPU
    createTextureImage();
    createTextureImageView();
    createTextureSampler();
    
    createDepthTextureImage();
    createDepthTextureImageView();
    createDepthTextureSampler();
    
    createNormalTextureImage();
    createNormalTextureImageView();
    createNormalTextureSampler();
    
    createAlbedoTextureImage();
    createAlbedoTextureImageView();
    createAlbedoTextureSampler();
    
    createMVTextureImage();
    createMVTextureImageView();
    createMVTextureSampler();
    
    createTNRResources();       // Temporal Noise Reduction resources
    createSNRResources();       // Spatial Noise Reduction resources
    
    createSNRResources();       // Spatial Noise Reduction resources
    createTNR2Resources();      // TNR2 resources
    createComputeFresnelResources(); // Compute Fresnel resources
    
    createDescriptorPool();     // Pool for allocating descriptor sets.
    createDescriptorSets();     // Allocate and update descriptor sets (bind images to shaders).
    createTNRDescriptorSets();
    createSNRDescriptorSets();
    createSNR2Resources();
    createSNR2DescriptorSets();
    createTNR2DescriptorSets();
    createComputeFresnelDescriptorSets();
    
    createSyncObjects();        // Create semaphores and fences for frame synchronization.
}

void VulkanRenderer::mainLoop() {
    while (!glfwWindowShouldClose(window)) {
        glfwPollEvents();
        drawFrame();
    }
    vkDeviceWaitIdle(device);
}

void VulkanRenderer::cleanup() {
    for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
        vkDestroySemaphore(device, renderFinishedSemaphores[i], nullptr);
        vkDestroySemaphore(device, imageAvailableSemaphores[i], nullptr);
        vkDestroyFence(device, inFlightFences[i], nullptr);
    }

    vkDestroyCommandPool(device, commandPool, nullptr);
    vkDestroyDescriptorPool(device, descriptorPool, nullptr);
    vkDestroySampler(device, textureSampler, nullptr);
    vkDestroyImageView(device, textureImageView, nullptr);
    vkDestroyImage(device, textureImage, nullptr);
    vkFreeMemory(device, textureImageMemory, nullptr);

    vkDestroySampler(device, depthTextureSampler, nullptr);
    vkDestroyImageView(device, depthTextureImageView, nullptr);
    vkDestroyImage(device, depthTextureImage, nullptr);
    vkFreeMemory(device, depthTextureImageMemory, nullptr);

    vkDestroySampler(device, normalTextureSampler, nullptr);
    vkDestroyImageView(device, normalTextureImageView, nullptr);
    vkDestroyImage(device, normalTextureImage, nullptr);
    vkFreeMemory(device, normalTextureImageMemory, nullptr);
    
    vkDestroySampler(device, albedoTextureSampler, nullptr);
    vkDestroyImageView(device, albedoTextureImageView, nullptr);
    vkDestroyImage(device, albedoTextureImage, nullptr);
    vkFreeMemory(device, albedoTextureImageMemory, nullptr);
    
    vkDestroyBuffer(device, stagingBuffer, nullptr);
    vkFreeMemory(device, stagingBufferMemory, nullptr);

    vkDestroyBuffer(device, depthStagingBuffer, nullptr);
    vkFreeMemory(device, depthStagingBufferMemory, nullptr);

    vkDestroyBuffer(device, normalStagingBuffer, nullptr);
    vkFreeMemory(device, normalStagingBufferMemory, nullptr);

    vkDestroyBuffer(device, albedoStagingBuffer, nullptr);
    vkFreeMemory(device, albedoStagingBufferMemory, nullptr);

    vkDestroyPipeline(device, depthDSPipeline, nullptr);
    vkDestroyPipelineLayout(device, depthDSPipelineLayout, nullptr);
    vkDestroyDescriptorSetLayout(device, depthDSDescriptorSetLayout, nullptr);
    vkDestroyFramebuffer(device, depthDSFramebuffer, nullptr);
    vkDestroyRenderPass(device, depthDSRenderPass, nullptr);
    vkDestroyImageView(device, depthDSImageView, nullptr);
    vkDestroyImage(device, depthDSImage, nullptr);
    vkFreeMemory(device, depthDSImageMemory, nullptr);

    vkDestroyPipeline(device, offscreenPipeline, nullptr);
    vkDestroyPipelineLayout(device, offscreenPipelineLayout, nullptr);
    vkDestroyFramebuffer(device, offscreenFramebuffer, nullptr);
    vkDestroyRenderPass(device, offscreenRenderPass, nullptr);
    vkDestroyImageView(device, offscreenImageView, nullptr);
    vkDestroyImage(device, offscreenImage, nullptr);
    vkFreeMemory(device, offscreenImageMemory, nullptr);
    vkDestroySampler(device, offscreenSampler, nullptr);

    vkDestroyPipeline(device, tnrPipeline, nullptr);
    vkDestroyPipelineLayout(device, tnrPipelineLayout, nullptr);
    vkDestroyRenderPass(device, tnrRenderPass, nullptr);
    vkDestroyDescriptorSetLayout(device, tnrDescriptorSetLayout, nullptr);
    
    vkDestroyImageView(device, tnrIntermediateColorImageView, nullptr);
    vkDestroyImage(device, tnrIntermediateColorImage, nullptr);
    vkFreeMemory(device, tnrIntermediateColorImageMemory, nullptr);

    vkDestroyImageView(device, tnrOut2ImageView, nullptr);
    vkDestroyImage(device, tnrOut2Image, nullptr);
    vkFreeMemory(device, tnrOut2ImageMemory, nullptr);

    for (int i = 0; i < 2; i++) {
        vkDestroyFramebuffer(device, tnrFramebuffers[i], nullptr);
        vkDestroyImageView(device, tnrInfoImageViews[i], nullptr);
        vkDestroyImage(device, tnrInfoImages[i], nullptr);
        vkFreeMemory(device, tnrInfoImageMemories[i], nullptr);
    }

    vkDestroyPipeline(device, snrPipeline, nullptr);
    vkDestroyPipelineLayout(device, snrPipelineLayout, nullptr);
    vkDestroyRenderPass(device, snrRenderPass, nullptr);
    vkDestroyDescriptorSetLayout(device, snrDescriptorSetLayout, nullptr);
    for (int i = 0; i < 2; i++) {
        vkDestroyFramebuffer(device, snrFramebuffers[i], nullptr);
        vkDestroyImageView(device, snrImageViews[i], nullptr);
        vkDestroyImage(device, snrImages[i], nullptr);
        vkFreeMemory(device, snrImageMemories[i], nullptr);
    }

    vkDestroyPipeline(device, snr2Pipeline, nullptr);
    vkDestroyPipelineLayout(device, snr2PipelineLayout, nullptr);
    vkDestroyRenderPass(device, snr2RenderPass, nullptr);
    vkDestroyDescriptorSetLayout(device, snr2DescriptorSetLayout, nullptr);
    for (int i = 0; i < 2; i++) {
        vkDestroyFramebuffer(device, snr2Framebuffers[i], nullptr);
        vkDestroyImageView(device, snr2ImageViews[i], nullptr);
        vkDestroyImage(device, snr2Images[i], nullptr);
        vkFreeMemory(device, snr2ImageMemories[i], nullptr);
    }

    vkDestroySampler(device, mvTextureSampler, nullptr);
    vkDestroyImageView(device, mvTextureImageView, nullptr);
    vkDestroyImage(device, mvTextureImage, nullptr);
    vkFreeMemory(device, mvTextureImageMemory, nullptr);
    vkDestroyBuffer(device, mvStagingBuffer, nullptr);
    vkFreeMemory(device, mvStagingBufferMemory, nullptr);

    vkDestroyPipeline(device, finalPipeline, nullptr);
    vkDestroyPipelineLayout(device, finalPipelineLayout, nullptr);
    vkDestroyDescriptorSetLayout(device, finalDescriptorSetLayout, nullptr);

    for (auto framebuffer : swapchainFramebuffers) {
        vkDestroyFramebuffer(device, framebuffer, nullptr);
    }

    vkDestroyRenderPass(device, renderPass, nullptr);

    for (auto imageView : swapchainImageViews) {
        vkDestroyImageView(device, imageView, nullptr);
    }

    vkDestroySwapchainKHR(device, swapchain, nullptr);
    vkDestroyDescriptorSetLayout(device, descriptorSetLayout, nullptr);
    vkDestroyDevice(device, nullptr);

    if (enableValidationLayers) {
        DestroyDebugUtilsMessengerEXT(instance, debugMessenger, nullptr);
    }

    vkDestroySurfaceKHR(instance, surface, nullptr);
    vkDestroyInstance(instance, nullptr);
    glfwDestroyWindow(window);
    glfwTerminate();
}

// 1. Create the Vulkan Instance.
// This initializes the Vulkan library and allows the application to pass information about itself to the driver.
void VulkanRenderer::createInstance() {
    // If we are in Debug mode, check if the validation layers are actually installed on the system.
    if (enableValidationLayers && !checkValidationLayerSupport()) {
        throw std::runtime_error("validation layers requested, but not available!");
    }

    // Optional struct to provide info about our app to the driver (mostly for internal driver statistics).
    VkApplicationInfo appInfo{};
    appInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    appInfo.pApplicationName = "Vulkan Image Player";
    appInfo.applicationVersion = VK_MAKE_VERSION(1, 0, 0);
    appInfo.pEngineName = "No Engine";
    appInfo.engineVersion = VK_MAKE_VERSION(1, 0, 0);
    appInfo.apiVersion = VK_API_VERSION_1_0;

    // Main creation info struct.
    VkInstanceCreateInfo createInfo{};
    createInfo.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    createInfo.pApplicationInfo = &appInfo;

    // Get the extensions needed by GLFW (to draw to a window) + any debug extensions.
    auto extensions = getRequiredExtensions();
    createInfo.enabledExtensionCount = static_cast<uint32_t>(extensions.size());
    createInfo.ppEnabledExtensionNames = extensions.data();

    // Configure validation layers if enabled.
    VkDebugUtilsMessengerCreateInfoEXT debugCreateInfo{};
    if (enableValidationLayers) {
        createInfo.enabledLayerCount = static_cast<uint32_t>(validationLayers.size());
        createInfo.ppEnabledLayerNames = validationLayers.data();

        // Also setup a debug messenger for the creation and destruction of the instance itself (which happens before/after the main messenger exists).
        debugCreateInfo.sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT;
        debugCreateInfo.messageSeverity = VK_DEBUG_UTILS_MESSAGE_SEVERITY_VERBOSE_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT;
        debugCreateInfo.messageType = VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT;
        debugCreateInfo.pfnUserCallback = debugCallback; // Function to call when an error happens
        createInfo.pNext = (VkDebugUtilsMessengerCreateInfoEXT*) &debugCreateInfo;
    } else {
        createInfo.enabledLayerCount = 0;
        createInfo.pNext = nullptr;
    }

    // Special fix for macOS (MoltenVK) which requires the Portability Enumeration extension.
    for (const char* extName : extensions) {
        if (strcmp(extName, VK_KHR_PORTABILITY_ENUMERATION_EXTENSION_NAME) == 0) {
            createInfo.flags |= VK_INSTANCE_CREATE_ENUMERATE_PORTABILITY_BIT_KHR;
            break;
        }
    }

    if (vkCreateInstance(&createInfo, nullptr, &instance) != VK_SUCCESS) {
        throw std::runtime_error("failed to create instance!");
    }
}

// 2. Setup the debug messenger to capture validation errors.
void VulkanRenderer::setupDebugMessenger() {
    if (!enableValidationLayers) return;

    VkDebugUtilsMessengerCreateInfoEXT createInfo{};
    createInfo.sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT;
    // Capture Warnings and Errors (and Verbose info if you want everything)
    createInfo.messageSeverity = VK_DEBUG_UTILS_MESSAGE_SEVERITY_VERBOSE_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT;
    createInfo.messageType = VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT;
    createInfo.pfnUserCallback = debugCallback;

    if (CreateDebugUtilsMessengerEXT(instance, &createInfo, nullptr, &debugMessenger) != VK_SUCCESS) {
        throw std::runtime_error("failed to set up debug messenger!");
    }
}

// 3. Create a Surface (the connection between the Window system and Vulkan).
void VulkanRenderer::createSurface() {
    if (glfwCreateWindowSurface(instance, window, nullptr, &surface) != VK_SUCCESS) {
        throw std::runtime_error("failed to create window surface!");
    }
}

// 4. Select a Physical Device (GPU) that supports the features we need.
void VulkanRenderer::pickPhysicalDevice() {
    uint32_t deviceCount = 0;
    vkEnumeratePhysicalDevices(instance, &deviceCount, nullptr);
    if (deviceCount == 0) throw std::runtime_error("failed to find GPUs with Vulkan support!");

    std::vector<VkPhysicalDevice> devices(deviceCount);
    vkEnumeratePhysicalDevices(instance, &deviceCount, devices.data());

    // Iterate over available devices and pick the first one that works.
    // In a real app, you might rate them (e.g., prefer Discrete GPU over Integrated).
    for (const auto& device : devices) {
        physicalDevice = device; // Just pick the first one for simplicity
        break; 
    }

    if (physicalDevice == VK_NULL_HANDLE) {
        throw std::runtime_error("failed to find a suitable GPU!");
    }
}

// 5. Create a Logical Device (interface to the physical GPU).
// This is where we specify which Queues we want to use (e.g., Graphics queue).
void VulkanRenderer::createLogicalDevice() {
    // Find queue families (types of queues supported by the GPU).
    uint32_t queueFamilyCount = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(physicalDevice, &queueFamilyCount, nullptr);
    std::vector<VkQueueFamilyProperties> queueFamilies(queueFamilyCount);
    vkGetPhysicalDeviceQueueFamilyProperties(physicalDevice, &queueFamilyCount, queueFamilies.data());

    // Look for a queue family that supports Graphics operations.
    int graphicsFamily = -1;
    for (int i = 0; i < queueFamilies.size(); i++) {
        if (queueFamilies[i].queueFlags & VK_QUEUE_GRAPHICS_BIT) {
            graphicsFamily = i;
            break;
        }
    }
    graphicsQueueFamilyIndex = graphicsFamily;
    
    // Structure to specify we want 1 queue from that family.
    float queuePriority = 1.0f;
    VkDeviceQueueCreateInfo queueCreateInfo{};
    queueCreateInfo.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
    queueCreateInfo.queueFamilyIndex = graphicsFamily;
    queueCreateInfo.queueCount = 1;
    queueCreateInfo.pQueuePriorities = &queuePriority;

    // Device features we want to enable (e.g., geometry shaders). Leaving empty for now.
    VkPhysicalDeviceFeatures deviceFeatures{};

    std::vector<const char*> enabledExtensions = deviceExtensions;
    
    // Check for "portability subset" again (macOS requirement).
    uint32_t extensionCount;
    vkEnumerateDeviceExtensionProperties(physicalDevice, nullptr, &extensionCount, nullptr);
    std::vector<VkExtensionProperties> availableExtensions(extensionCount);
    vkEnumerateDeviceExtensionProperties(physicalDevice, nullptr, &extensionCount, availableExtensions.data());
    
    for (const auto& extension : availableExtensions) {
        if (strcmp(extension.extensionName, "VK_KHR_portability_subset") == 0) {
            enabledExtensions.push_back("VK_KHR_portability_subset");
            break;
        }
    }

    // Main Logical Device creation info.
    VkDeviceCreateInfo createInfo{};
    createInfo.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    createInfo.pQueueCreateInfos = &queueCreateInfo;
    createInfo.queueCreateInfoCount = 1;
    createInfo.pEnabledFeatures = &deviceFeatures;
    createInfo.enabledExtensionCount = static_cast<uint32_t>(enabledExtensions.size());
    createInfo.ppEnabledExtensionNames = enabledExtensions.data();

    // Enable validation layers on the device too (legacy but good practice).
    if (enableValidationLayers) {
        createInfo.enabledLayerCount = static_cast<uint32_t>(validationLayers.size());
        createInfo.ppEnabledLayerNames = validationLayers.data();
    } else {
        createInfo.enabledLayerCount = 0;
    }

    if (vkCreateDevice(physicalDevice, &createInfo, nullptr, &device) != VK_SUCCESS) {
        throw std::runtime_error("failed to create logical device!");
    }

    // Retrieve the handle to the execution queue from the logical device.
    vkGetDeviceQueue(device, graphicsFamily, 0, &graphicsQueue);
    vkGetDeviceQueue(device, graphicsFamily, 0, &presentQueue); // Assuming same family for simplicity
}

// 6. Create the Swapchain.
// The swapchain is a queue of images that are waiting to be presented to the screen.
// It acts as a buffer between the GPU drawing and the fast-refreshing monitor.
void VulkanRenderer::createSwapchain() {
    VkSwapchainCreateInfoKHR createInfo{};
    createInfo.sType = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR;
    createInfo.surface = surface;
    createInfo.minImageCount = 3; // Triple buffering (reduces tearing and stuttering)
    createInfo.imageFormat = VK_FORMAT_B8G8R8A8_SRGB; // Standard color format (Blue, Green, Red, Alpha)
    createInfo.imageColorSpace = VK_COLOR_SPACE_SRGB_NONLINEAR_KHR;
    createInfo.imageExtent = {WIDTH, HEIGHT}; // Resolution
    createInfo.imageArrayLayers = 1; // Always 1 for 2D images
    createInfo.imageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT; // We will render directly to these images
    createInfo.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE; // Only one queue needs access
    createInfo.preTransform = VK_SURFACE_TRANSFORM_IDENTITY_BIT_KHR; // Don't rotate/flip the image on presentation
    createInfo.compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR; // Ignore alpha channel for the window composition (no transparent windows)
    createInfo.presentMode = VK_PRESENT_MODE_FIFO_KHR; // Vsync enabled (FIFO = First In First Out)
    createInfo.clipped = VK_TRUE; // Don't process pixels covered by other windows

    if (vkCreateSwapchainKHR(device, &createInfo, nullptr, &swapchain) != VK_SUCCESS) {
        throw std::runtime_error("failed to create swap chain!");
    }

    // Retrieve the actual images created by the swapchain extension
    vkGetSwapchainImagesKHR(device, swapchain, &createInfo.minImageCount, nullptr);
    swapchainImages.resize(createInfo.minImageCount);
    vkGetSwapchainImagesKHR(device, swapchain, &createInfo.minImageCount, swapchainImages.data());

    swapchainImageFormat = createInfo.imageFormat;
    swapchainExtent = createInfo.imageExtent;
}

// 7. Create Image Views for the Swapchain Images.
// Vulkan pipelines don't access Images directly; they access "Image Views" which describe *how* to access the image data.
void VulkanRenderer::createImageViews() {
    swapchainImageViews.resize(swapchainImages.size());
    for (size_t i = 0; i < swapchainImages.size(); i++) {
        swapchainImageViews[i] = createImageView(swapchainImages[i], swapchainImageFormat);
    }
}

// 8. Create Render Passes.
// A Render Pass tells Vulkan about the attachments (images) we will be using during a drawing operation.
// It describes formats, sample counts, and what to do with the data at the start/end of the pass (Load/Store ops).
void VulkanRenderer::createRenderPass() {
    // === Main Render Pass (For presenting to screen) ===
    VkAttachmentDescription colorAttachment{};
    colorAttachment.format = swapchainImageFormat;
    colorAttachment.samples = VK_SAMPLE_COUNT_1_BIT; // No MSAA for now
    colorAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR; // Clear screen to black before drawing
    colorAttachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE; // Save the drawn content so it can be displayed
    colorAttachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    colorAttachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    colorAttachment.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED; // We don't care what was here before
    colorAttachment.finalLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR; // Ready to be presented to swapchain

    VkAttachmentReference colorAttachmentRef{};
    colorAttachmentRef.attachment = 0; // Index in the pAttachments array
    colorAttachmentRef.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL; // Optimized layout for writing color

    VkSubpassDescription subpass{};
    subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
    subpass.colorAttachmentCount = 1;
    subpass.pColorAttachments = &colorAttachmentRef;

    VkRenderPassCreateInfo renderPassInfo{};
    renderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
    renderPassInfo.attachmentCount = 1;
    renderPassInfo.pAttachments = &colorAttachment;
    renderPassInfo.subpassCount = 1;
    renderPassInfo.pSubpasses = &subpass;

    if (vkCreateRenderPass(device, &renderPassInfo, nullptr, &renderPass) != VK_SUCCESS) {
        throw std::runtime_error("failed to create render pass!");
    }

    // === Offscreen Render Pass (For Ray Marching) ===
    // This renders to an intermediate texture, not the screen.
    VkAttachmentDescription offscreenAttachment{};
    offscreenAttachment.format = VK_FORMAT_R16G16B16A16_SFLOAT;
    offscreenAttachment.samples = VK_SAMPLE_COUNT_1_BIT;
    offscreenAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
    offscreenAttachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    offscreenAttachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    offscreenAttachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    offscreenAttachment.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    offscreenAttachment.finalLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL; // Allows us to sample it in the next shader

    VkAttachmentReference offscreenAttachmentRef{};
    offscreenAttachmentRef.attachment = 0;
    offscreenAttachmentRef.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

    VkSubpassDescription offscreenSubpass{};
    offscreenSubpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
    offscreenSubpass.colorAttachmentCount = 1;
    offscreenSubpass.pColorAttachments = &offscreenAttachmentRef;

    // Dependency to ensure previous reads are finished before we write to this image.
    VkSubpassDependency offscreenDependency{};
    offscreenDependency.srcSubpass = VK_SUBPASS_EXTERNAL;
    offscreenDependency.dstSubpass = 0;
    offscreenDependency.srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
    offscreenDependency.srcAccessMask = 0;
    offscreenDependency.dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
    offscreenDependency.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;

    VkRenderPassCreateInfo offscreenRenderPassInfo{};
    offscreenRenderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
    offscreenRenderPassInfo.attachmentCount = 1;
    offscreenRenderPassInfo.pAttachments = &offscreenAttachment;
    offscreenRenderPassInfo.subpassCount = 1;
    offscreenRenderPassInfo.pSubpasses = &offscreenSubpass;
    offscreenRenderPassInfo.dependencyCount = 1;
    offscreenRenderPassInfo.pDependencies = &offscreenDependency;

    if (vkCreateRenderPass(device, &offscreenRenderPassInfo, nullptr, &offscreenRenderPass) != VK_SUCCESS) {
        throw std::runtime_error("failed to create offscreen render pass!");
    }

    // === DepthDS Render Pass (Depth Downsampling) ===
    VkAttachmentDescription dsAttachment{};
    dsAttachment.format = VK_FORMAT_R16G16B16A16_SFLOAT;
    dsAttachment.samples = VK_SAMPLE_COUNT_1_BIT;
    dsAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
    dsAttachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    dsAttachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    dsAttachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    dsAttachment.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    dsAttachment.finalLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

    VkAttachmentReference dsColorAttachmentRef{};
    dsColorAttachmentRef.attachment = 0;
    dsColorAttachmentRef.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

    VkSubpassDescription dsSubpass{};
    dsSubpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
    dsSubpass.colorAttachmentCount = 1;
    dsSubpass.pColorAttachments = &dsColorAttachmentRef;

    VkSubpassDependency dsDependency{};
    dsDependency.srcSubpass = VK_SUBPASS_EXTERNAL;
    dsDependency.dstSubpass = 0;
    dsDependency.srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
    dsDependency.srcAccessMask = 0;
    dsDependency.dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
    dsDependency.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;

    VkRenderPassCreateInfo dsRenderPassInfo{};
    dsRenderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
    dsRenderPassInfo.attachmentCount = 1;
    dsRenderPassInfo.pAttachments = &dsAttachment;
    dsRenderPassInfo.subpassCount = 1;
    dsRenderPassInfo.pSubpasses = &dsSubpass;
    dsRenderPassInfo.dependencyCount = 1;
    dsRenderPassInfo.pDependencies = &dsDependency;

    if (vkCreateRenderPass(device, &dsRenderPassInfo, nullptr, &depthDSRenderPass) != VK_SUCCESS) {
        throw std::runtime_error("failed to create depthDS render pass!");
    }
}

// 9. Create Descriptor Set Layouts.
// A Descriptor Set Layout is the interface between the code and the shader.
// It defines what kind of buffers/images the shader expects (e.g., "Binding 0 is a Texture").
void VulkanRenderer::createDescriptorSetLayout() {
    VkDescriptorSetLayoutBinding samplerLayoutBinding{};
    samplerLayoutBinding.binding = 0; // Matches `layout(binding = 0)` in shader
    samplerLayoutBinding.descriptorCount = 1;
    samplerLayoutBinding.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    samplerLayoutBinding.pImmutableSamplers = nullptr;
    samplerLayoutBinding.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT; // Only the fragment shader uses this

    VkDescriptorSetLayoutBinding depthSamplerLayoutBinding{};
    depthSamplerLayoutBinding.binding = 1;
    depthSamplerLayoutBinding.descriptorCount = 1;
    depthSamplerLayoutBinding.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    depthSamplerLayoutBinding.pImmutableSamplers = nullptr;
    depthSamplerLayoutBinding.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;

    VkDescriptorSetLayoutBinding normalSamplerLayoutBinding{};
    normalSamplerLayoutBinding.binding = 2;
    normalSamplerLayoutBinding.descriptorCount = 1;
    normalSamplerLayoutBinding.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    normalSamplerLayoutBinding.pImmutableSamplers = nullptr;
    normalSamplerLayoutBinding.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;

    std::vector<VkDescriptorSetLayoutBinding> bindings = {samplerLayoutBinding, depthSamplerLayoutBinding, normalSamplerLayoutBinding};

    VkDescriptorSetLayoutCreateInfo layoutInfo{};
    layoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    layoutInfo.bindingCount = static_cast<uint32_t>(bindings.size());
    layoutInfo.pBindings = bindings.data();

    if (vkCreateDescriptorSetLayout(device, &layoutInfo, nullptr, &descriptorSetLayout) != VK_SUCCESS) {
        throw std::runtime_error("failed to create descriptor set layout!");
    }

    // DepthDS Descriptor Set Layout
    VkDescriptorSetLayoutBinding dsDepthBinding{};
    dsDepthBinding.binding = 0;
    dsDepthBinding.descriptorCount = 1;
    dsDepthBinding.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    dsDepthBinding.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;

    VkDescriptorSetLayoutBinding dsAlbedoBinding{};
    dsAlbedoBinding.binding = 1;
    dsAlbedoBinding.descriptorCount = 1;
    dsAlbedoBinding.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    dsAlbedoBinding.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;

    VkDescriptorSetLayoutBinding dsNormalBinding{};
    dsNormalBinding.binding = 2;
    dsNormalBinding.descriptorCount = 1;
    dsNormalBinding.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    dsNormalBinding.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;

    VkDescriptorSetLayoutBinding dsNewAlbedoBinding{};
    dsNewAlbedoBinding.binding = 3;
    dsNewAlbedoBinding.descriptorCount = 1;
    dsNewAlbedoBinding.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    dsNewAlbedoBinding.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;

    std::vector<VkDescriptorSetLayoutBinding> dsBindings = {dsDepthBinding, dsAlbedoBinding, dsNormalBinding, dsNewAlbedoBinding};
    VkDescriptorSetLayoutCreateInfo dsLayoutInfo{};
    dsLayoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    dsLayoutInfo.bindingCount = static_cast<uint32_t>(dsBindings.size());
    dsLayoutInfo.pBindings = dsBindings.data();

    if (vkCreateDescriptorSetLayout(device, &dsLayoutInfo, nullptr, &depthDSDescriptorSetLayout) != VK_SUCCESS) {
        throw std::runtime_error("failed to create depthDS descriptor set layout!");
    }
}

// 10. Create Graphics Pipelines.
// A pipeline combines Shaders + Fixed Function states (rasterizer, blending, depth test, viewport).
// Once created, these states are immutable (you can't change them without creating a new pipeline).
void VulkanRenderer::createGraphicsPipeline() {
    // Load compiled shader code (SPIR-V binary)
    auto vertShaderCode = readFile(std::string(SHADER_DIR) + "/shader.vert.spv");
    auto rmFragShaderCode = readFile(std::string(SHADER_DIR) + "/RM.frag.spv");
    auto drawFragShaderCode = readFile(std::string(SHADER_DIR) + "/draw.frag.spv");
    auto dsFragShaderCode = readFile(std::string(SHADER_DIR) + "/depthDS.frag.spv");

    VkShaderModule vertShaderModule = createShaderModule(vertShaderCode);
    VkShaderModule rmFragShaderModule = createShaderModule(rmFragShaderCode);
    VkShaderModule drawFragShaderModule = createShaderModule(drawFragShaderCode);
    VkShaderModule dsFragShaderModule = createShaderModule(dsFragShaderCode);

    // === RM Pipeline (Offscreen Ray Marching) ===
    VkPipelineShaderStageCreateInfo rmVertShaderStageInfo{};
    rmVertShaderStageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    rmVertShaderStageInfo.stage = VK_SHADER_STAGE_VERTEX_BIT;
    rmVertShaderStageInfo.module = vertShaderModule;
    rmVertShaderStageInfo.pName = "main";

    VkPipelineShaderStageCreateInfo rmFragShaderStageInfo{};
    rmFragShaderStageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    rmFragShaderStageInfo.stage = VK_SHADER_STAGE_FRAGMENT_BIT;
    rmFragShaderStageInfo.module = rmFragShaderModule;
    rmFragShaderStageInfo.pName = "main";

    VkPipelineShaderStageCreateInfo rmShaderStages[] = {rmVertShaderStageInfo, rmFragShaderStageInfo};

    // Vertex Input: How data is passed from vertex buffers to the vertex shader.
    // We are generating a fullscreen triangle in code, so we don't need input buffers here.
    VkPipelineVertexInputStateCreateInfo vertexInputInfo{};
    vertexInputInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
    vertexInputInfo.vertexBindingDescriptionCount = 0;
    vertexInputInfo.vertexAttributeDescriptionCount = 0;

    // Input Assembly: How vertices are assembled into primitives (Triangles).
    VkPipelineInputAssemblyStateCreateInfo inputAssembly{};
    inputAssembly.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
    inputAssembly.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
    inputAssembly.primitiveRestartEnable = VK_FALSE;

    // Viewport & Scissor: Defines the region of the framebuffer to render to.
    VkPipelineViewportStateCreateInfo viewportState{};
    viewportState.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
    viewportState.viewportCount = 1;
    viewportState.scissorCount = 1;

    // Rasterizer: Turns geometry into fragments (pixels).
    VkPipelineRasterizationStateCreateInfo rasterizer{};
    rasterizer.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
    rasterizer.depthClampEnable = VK_FALSE;
    rasterizer.rasterizerDiscardEnable = VK_FALSE;
    rasterizer.polygonMode = VK_POLYGON_MODE_FILL; // Fill the triangle (solid)
    rasterizer.lineWidth = 1.0f;
    rasterizer.cullMode = VK_CULL_MODE_BACK_BIT; // Don't draw back-facing triangles
    rasterizer.frontFace = VK_FRONT_FACE_CLOCKWISE;
    rasterizer.depthBiasEnable = VK_FALSE;

    // Multisampling: Anti-aliasing (disabled here).
    VkPipelineMultisampleStateCreateInfo multisampling{};
    multisampling.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
    multisampling.sampleShadingEnable = VK_FALSE;
    multisampling.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

    // Color Blending: How to mix new pixel colors with existing ones.
    VkPipelineColorBlendAttachmentState colorBlendAttachment{};
    colorBlendAttachment.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
    colorBlendAttachment.blendEnable = VK_FALSE; // Overwrite existing color

    VkPipelineColorBlendStateCreateInfo colorBlending{};
    colorBlending.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
    colorBlending.logicOpEnable = VK_FALSE;
    colorBlending.attachmentCount = 1;
    colorBlending.pAttachments = &colorBlendAttachment;

    // Dynamic State: States that CAN be changed without recreating the pipeline (e.g., resizing viewport).
    std::vector<VkDynamicState> dynamicStates = {
        VK_DYNAMIC_STATE_VIEWPORT,
        VK_DYNAMIC_STATE_SCISSOR
    };
    VkPipelineDynamicStateCreateInfo dynamicState{};
    dynamicState.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
    dynamicState.dynamicStateCount = static_cast<uint32_t>(dynamicStates.size());
    dynamicState.pDynamicStates = dynamicStates.data();

    // Offscreen Pipeline Layout (connects descriptor layouts to pipeline)
    VkPipelineLayoutCreateInfo offscreenPipelineLayoutInfo{};
    offscreenPipelineLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    offscreenPipelineLayoutInfo.setLayoutCount = 1;
    offscreenPipelineLayoutInfo.pSetLayouts = &descriptorSetLayout;

    if (vkCreatePipelineLayout(device, &offscreenPipelineLayoutInfo, nullptr, &offscreenPipelineLayout) != VK_SUCCESS) {
        throw std::runtime_error("failed to create offscreen pipeline layout!");
    }

    VkGraphicsPipelineCreateInfo offscreenPipelineInfo{};
    offscreenPipelineInfo.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
    offscreenPipelineInfo.stageCount = 2;
    offscreenPipelineInfo.pStages = rmShaderStages;
    offscreenPipelineInfo.pVertexInputState = &vertexInputInfo;
    offscreenPipelineInfo.pInputAssemblyState = &inputAssembly;
    offscreenPipelineInfo.pViewportState = &viewportState;
    offscreenPipelineInfo.pRasterizationState = &rasterizer;
    offscreenPipelineInfo.pMultisampleState = &multisampling;
    offscreenPipelineInfo.pColorBlendState = &colorBlending;
    offscreenPipelineInfo.pDynamicState = &dynamicState;
    offscreenPipelineInfo.layout = offscreenPipelineLayout;
    offscreenPipelineInfo.renderPass = offscreenRenderPass;
    offscreenPipelineInfo.subpass = 0;

    if (vkCreateGraphicsPipelines(device, VK_NULL_HANDLE, 1, &offscreenPipelineInfo, nullptr, &offscreenPipeline) != VK_SUCCESS) {
        throw std::runtime_error("failed to create offscreen graphics pipeline!");
    }

    // === DepthDS Pipeline ===
    VkPipelineShaderStageCreateInfo dsFragShaderStageInfo{};
    dsFragShaderStageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    dsFragShaderStageInfo.stage = VK_SHADER_STAGE_FRAGMENT_BIT;
    dsFragShaderStageInfo.module = dsFragShaderModule;
    dsFragShaderStageInfo.pName = "main";

    VkPipelineShaderStageCreateInfo dsShaderStages[] = {rmVertShaderStageInfo, dsFragShaderStageInfo};

    VkPipelineLayoutCreateInfo dsPipelineLayoutInfo{};
    dsPipelineLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    dsPipelineLayoutInfo.setLayoutCount = 1;
    dsPipelineLayoutInfo.pSetLayouts = &depthDSDescriptorSetLayout;

    if (vkCreatePipelineLayout(device, &dsPipelineLayoutInfo, nullptr, &depthDSPipelineLayout) != VK_SUCCESS) {
        throw std::runtime_error("failed to create depthDS pipeline layout!");
    }

    VkGraphicsPipelineCreateInfo dsPipelineInfo = offscreenPipelineInfo;
    dsPipelineInfo.pStages = dsShaderStages;
    dsPipelineInfo.layout = depthDSPipelineLayout;
    dsPipelineInfo.renderPass = depthDSRenderPass;

    if (vkCreateGraphicsPipelines(device, VK_NULL_HANDLE, 1, &dsPipelineInfo, nullptr, &depthDSPipeline) != VK_SUCCESS) {
        throw std::runtime_error("failed to create depthDS graphics pipeline!");
    }

    // === Final (Upscale) Pipeline ===
    VkPipelineShaderStageCreateInfo finalFragShaderStageInfo{};
    finalFragShaderStageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    finalFragShaderStageInfo.stage = VK_SHADER_STAGE_FRAGMENT_BIT;
    finalFragShaderStageInfo.module = drawFragShaderModule;
    finalFragShaderStageInfo.pName = "main";

    VkPipelineShaderStageCreateInfo finalShaderStages[] = {rmVertShaderStageInfo, finalFragShaderStageInfo};

    VkPipelineLayoutCreateInfo finalPipelineLayoutInfo{};
    finalPipelineLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    finalPipelineLayoutInfo.setLayoutCount = 1;
    finalPipelineLayoutInfo.pSetLayouts = &finalDescriptorSetLayout;

    if (vkCreatePipelineLayout(device, &finalPipelineLayoutInfo, nullptr, &finalPipelineLayout) != VK_SUCCESS) {
        throw std::runtime_error("failed to create final pipeline layout!");
    }

    VkGraphicsPipelineCreateInfo finalPipelineInfo = offscreenPipelineInfo;
    finalPipelineInfo.pStages = finalShaderStages;
    finalPipelineInfo.layout = finalPipelineLayout;
    finalPipelineInfo.renderPass = renderPass;

    if (vkCreateGraphicsPipelines(device, VK_NULL_HANDLE, 1, &finalPipelineInfo, nullptr, &finalPipeline) != VK_SUCCESS) {
        throw std::runtime_error("failed to create final graphics pipeline!");
    }

    vkDestroyShaderModule(device, dsFragShaderModule, nullptr);
    vkDestroyShaderModule(device, drawFragShaderModule, nullptr);
    vkDestroyShaderModule(device, rmFragShaderModule, nullptr);
    vkDestroyShaderModule(device, vertShaderModule, nullptr);
}

// 11. Create Framebuffers.
// A Framebuffer connects the actual Image Views (resources) to the Render Pass attachments (slots).
void VulkanRenderer::createFramebuffers() {
    swapchainFramebuffers.resize(swapchainImageViews.size());

    for (size_t i = 0; i < swapchainImageViews.size(); i++) {
        VkImageView attachments[] = {
            swapchainImageViews[i] // The image we want to draw to
        };

        VkFramebufferCreateInfo framebufferInfo{};
        framebufferInfo.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
        framebufferInfo.renderPass = renderPass; // The render pass this framebuffer is compatible with
        framebufferInfo.attachmentCount = 1;
        framebufferInfo.pAttachments = attachments;
        framebufferInfo.width = swapchainExtent.width;
        framebufferInfo.height = swapchainExtent.height;
        framebufferInfo.layers = 1;

        if (vkCreateFramebuffer(device, &framebufferInfo, nullptr, &swapchainFramebuffers[i]) != VK_SUCCESS) {
            throw std::runtime_error("failed to create framebuffer!");
        }
    }

    // Offscreen Framebuffer (connects Offscreen Image View to Offscreen Render Pass)
    VkImageView offscreenAttachments[] = {
        offscreenImageView
    };

    VkFramebufferCreateInfo offscreenFramebufferInfo{};
    offscreenFramebufferInfo.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
    offscreenFramebufferInfo.renderPass = offscreenRenderPass;
    offscreenFramebufferInfo.attachmentCount = 1;
    offscreenFramebufferInfo.pAttachments = offscreenAttachments;
    offscreenFramebufferInfo.width = RM_WIDTH;
    offscreenFramebufferInfo.height = RM_HEIGHT;
    offscreenFramebufferInfo.layers = 1;

    if (vkCreateFramebuffer(device, &offscreenFramebufferInfo, nullptr, &offscreenFramebuffer) != VK_SUCCESS) {
        throw std::runtime_error("failed to create offscreen framebuffer!");
    }
}

// 12. Create Command Pool.
// Commands (like "Draw Triangle") are recorded into Command Buffers.
// Command Buffers are allocated from a Command Pool.
void VulkanRenderer::createCommandPool() {
    VkCommandPoolCreateInfo poolInfo{};
    poolInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    poolInfo.queueFamilyIndex = graphicsQueueFamilyIndex; // Commands will be submitted to the Graphics Queue
    poolInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT; // Allow resetting individual buffers

    if (vkCreateCommandPool(device, &poolInfo, nullptr, &commandPool) != VK_SUCCESS) {
        throw std::runtime_error("failed to create command pool!");
    }
}

// 13. Create Texture Image.
// This loads an image into CPU memory, creates a GPU image, and copies the data over.
void VulkanRenderer::createTextureImage() {
    VkDeviceSize imageSize = WIDTH * HEIGHT * 4;

    // Create a temporary "Staging Buffer" in CPU-visible memory.
    // GPU memory is often not directly accessible by the CPU, so we map this buffer, write to it, then copy.
    createBuffer(imageSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, stagingBuffer, stagingBufferMemory);

    // Load initial data (e.g. from file) into the staging buffer
    updateTexture();

    // Create the actual Image on the GPU (Fast local memory).
    createImage(WIDTH, HEIGHT, VK_FORMAT_R8G8B8A8_UNORM, VK_IMAGE_TILING_OPTIMAL, VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, textureImage, textureImageMemory);

    // Prepare image to receive data
    transitionImageLayout(textureImage, VK_FORMAT_R8G8B8A8_UNORM, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
    // Copy data from Staging Buffer to GPU Image
    copyBufferToImage(stagingBuffer, textureImage, WIDTH, HEIGHT);
    // Prepare image for reading by the shader
    transitionImageLayout(textureImage, VK_FORMAT_R8G8B8A8_UNORM, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
}

void VulkanRenderer::createTextureImageView() {
    textureImageView = createImageView(textureImage, VK_FORMAT_R8G8B8A8_UNORM);
}

// 14. Create Texture Sampler.
// A sampler tells the shader how to read the texture (filtering, wrapping, etc.).
void VulkanRenderer::createTextureSampler() {
    VkSamplerCreateInfo samplerInfo{};
    samplerInfo.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
    samplerInfo.magFilter = VK_FILTER_LINEAR; // Linear interpolation when close up (Blurs pixels)
    samplerInfo.minFilter = VK_FILTER_LINEAR; // Linear interpolation when far away
    samplerInfo.addressModeU = VK_SAMPLER_ADDRESS_MODE_REPEAT; // Repeat pattern if coordinate > 1.0
    samplerInfo.addressModeV = VK_SAMPLER_ADDRESS_MODE_REPEAT;
    samplerInfo.addressModeW = VK_SAMPLER_ADDRESS_MODE_REPEAT;
    samplerInfo.anisotropyEnable = VK_FALSE;
    samplerInfo.borderColor = VK_BORDER_COLOR_INT_OPAQUE_BLACK;
    samplerInfo.unnormalizedCoordinates = VK_FALSE; // UVs are 0..1, not 0..width
    samplerInfo.compareEnable = VK_FALSE;
    samplerInfo.compareOp = VK_COMPARE_OP_ALWAYS;
    samplerInfo.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;

    if (vkCreateSampler(device, &samplerInfo, nullptr, &textureSampler) != VK_SUCCESS) {
        throw std::runtime_error("failed to create texture sampler!");
    }
}

void VulkanRenderer::createDepthTextureImage() {
    VkDeviceSize imageSize = WIDTH * HEIGHT * 4;

    createBuffer(imageSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, depthStagingBuffer, depthStagingBufferMemory);

    createImage(WIDTH, HEIGHT, VK_FORMAT_R8G8B8A8_UNORM, VK_IMAGE_TILING_OPTIMAL, VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, depthTextureImage, depthTextureImageMemory);

    transitionImageLayout(depthTextureImage, VK_FORMAT_R8G8B8A8_UNORM, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
    // Initial data will be loaded in the first updateTexture call
    transitionImageLayout(depthTextureImage, VK_FORMAT_R8G8B8A8_UNORM, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
}

void VulkanRenderer::createDepthTextureImageView() {
    depthTextureImageView = createImageView(depthTextureImage, VK_FORMAT_R8G8B8A8_UNORM);
}

void VulkanRenderer::createDepthTextureSampler() {
    VkSamplerCreateInfo samplerInfo{};
    samplerInfo.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
    samplerInfo.magFilter = VK_FILTER_LINEAR;
    samplerInfo.minFilter = VK_FILTER_LINEAR;
    samplerInfo.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE; // Clamp to edge (don't repeat)
    samplerInfo.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    samplerInfo.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    samplerInfo.anisotropyEnable = VK_FALSE;
    samplerInfo.borderColor = VK_BORDER_COLOR_INT_OPAQUE_BLACK;
    samplerInfo.unnormalizedCoordinates = VK_FALSE;
    samplerInfo.compareEnable = VK_FALSE;
    samplerInfo.compareOp = VK_COMPARE_OP_ALWAYS;
    samplerInfo.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;

    if (vkCreateSampler(device, &samplerInfo, nullptr, &depthTextureSampler) != VK_SUCCESS) {
        throw std::runtime_error("failed to create depth texture sampler!");
    }
}

void VulkanRenderer::createOffscreenResources() {
    createImage(RM_WIDTH, RM_HEIGHT, VK_FORMAT_R16G16B16A16_SFLOAT, VK_IMAGE_TILING_OPTIMAL, VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, offscreenImage, offscreenImageMemory);
    offscreenImageView = createImageView(offscreenImage, VK_FORMAT_R16G16B16A16_SFLOAT);
    transitionImageLayout(offscreenImage, VK_FORMAT_R16G16B16A16_SFLOAT, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);

    VkSamplerCreateInfo samplerInfo{};
    samplerInfo.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
    samplerInfo.magFilter = VK_FILTER_LINEAR;
    samplerInfo.minFilter = VK_FILTER_LINEAR;
    samplerInfo.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    samplerInfo.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    samplerInfo.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    samplerInfo.anisotropyEnable = VK_FALSE;
    samplerInfo.borderColor = VK_BORDER_COLOR_INT_OPAQUE_BLACK;
    samplerInfo.unnormalizedCoordinates = VK_FALSE;
    samplerInfo.compareEnable = VK_FALSE;
    samplerInfo.compareOp = VK_COMPARE_OP_ALWAYS;
    samplerInfo.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;

    if (vkCreateSampler(device, &samplerInfo, nullptr, &offscreenSampler) != VK_SUCCESS) {
        throw std::runtime_error("failed to create offscreen sampler!");
    }
}

void VulkanRenderer::createFinalDescriptorSetLayout() {
    VkDescriptorSetLayoutBinding samplerLayoutBinding{};
    samplerLayoutBinding.binding = 0;
    samplerLayoutBinding.descriptorCount = 1;
    samplerLayoutBinding.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    samplerLayoutBinding.pImmutableSamplers = nullptr;
    samplerLayoutBinding.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;

    VkDescriptorSetLayoutBinding colorLayoutBinding{};
    colorLayoutBinding.binding = 1;
    colorLayoutBinding.descriptorCount = 1;
    colorLayoutBinding.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    colorLayoutBinding.pImmutableSamplers = nullptr;
    colorLayoutBinding.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;

    VkDescriptorSetLayoutBinding bindings[] = {samplerLayoutBinding, colorLayoutBinding};

    VkDescriptorSetLayoutCreateInfo layoutInfo{};
    layoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    layoutInfo.bindingCount = 2;
    layoutInfo.pBindings = bindings;

    if (vkCreateDescriptorSetLayout(device, &layoutInfo, nullptr, &finalDescriptorSetLayout) != VK_SUCCESS) {
        throw std::runtime_error("failed to create final descriptor set layout!");
    }
}

// 15. Create Descriptor Pool.
// Like a Command Pool, but for Descriptors (Shader Resource bindings).
void VulkanRenderer::createDescriptorPool() {
    std::vector<VkDescriptorPoolSize> poolSizes(1);
    poolSizes[0].type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    poolSizes[0].descriptorCount = 100; // Enough for all our frames and textures

    VkDescriptorPoolCreateInfo poolInfo{};
    poolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    poolInfo.poolSizeCount = static_cast<uint32_t>(poolSizes.size());
    poolInfo.pPoolSizes = poolSizes.data();
    poolInfo.maxSets = 100;

    if (vkCreateDescriptorPool(device, &poolInfo, nullptr, &descriptorPool) != VK_SUCCESS) {
        throw std::runtime_error("failed to create descriptor pool!");
    }
}

// 16. Create Descriptor Sets.
// This actually allocates the "Descriptor Sets" from the pool and points them to the specific resources (Texture Image, Sampler).
void VulkanRenderer::createDescriptorSets() {
    // We need one set per frame-in-flight to avoid race conditions.
    std::vector<VkDescriptorSetLayout> layouts(MAX_FRAMES_IN_FLIGHT, descriptorSetLayout);
    VkDescriptorSetAllocateInfo allocInfo{};
    allocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    allocInfo.descriptorPool = descriptorPool;
    allocInfo.descriptorSetCount = static_cast<uint32_t>(MAX_FRAMES_IN_FLIGHT);
    allocInfo.pSetLayouts = layouts.data();

    descriptorSets.resize(MAX_FRAMES_IN_FLIGHT);
    if (vkAllocateDescriptorSets(device, &allocInfo, descriptorSets.data()) != VK_SUCCESS) {
        throw std::runtime_error("failed to allocate descriptor sets!");
    }

    for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
        // Info about the Texture to bind to Binding 0
        VkDescriptorImageInfo imageInfo{};
        imageInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        imageInfo.imageView = textureImageView;
        imageInfo.sampler = textureSampler;

        // Info about the Depth Texture to bind to Binding 1
        VkDescriptorImageInfo depthImageInfo{};
        depthImageInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        depthImageInfo.imageView = depthTextureImageView;
        depthImageInfo.sampler = depthTextureSampler;

        // Info about the Normal Texture to bind to Binding 2
        VkDescriptorImageInfo normalImageInfo{};
        normalImageInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        normalImageInfo.imageView = normalTextureImageView;
        normalImageInfo.sampler = normalTextureSampler;

        std::vector<VkWriteDescriptorSet> descriptorWrites(3);

        // Binding 0: Texture
        descriptorWrites[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        descriptorWrites[0].dstSet = descriptorSets[i];
        descriptorWrites[0].dstBinding = 0;
        descriptorWrites[0].dstArrayElement = 0;
        descriptorWrites[0].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        descriptorWrites[0].descriptorCount = 1;
        descriptorWrites[0].pImageInfo = &imageInfo;

        // Binding 1: Depth
        descriptorWrites[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        descriptorWrites[1].dstSet = descriptorSets[i];
        descriptorWrites[1].dstBinding = 1;
        descriptorWrites[1].dstArrayElement = 0;
        descriptorWrites[1].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        descriptorWrites[1].descriptorCount = 1;
        descriptorWrites[1].pImageInfo = &depthImageInfo;

        // Binding 2: Normal
        descriptorWrites[2].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        descriptorWrites[2].dstSet = descriptorSets[i];
        descriptorWrites[2].dstBinding = 2;
        descriptorWrites[2].dstArrayElement = 0;
        descriptorWrites[2].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        descriptorWrites[2].descriptorCount = 1;
        descriptorWrites[2].pImageInfo = &normalImageInfo;

        // Execute the write to update the descriptor set on the GPU
        vkUpdateDescriptorSets(device, static_cast<uint32_t>(descriptorWrites.size()), descriptorWrites.data(), 0, nullptr);
    }
    
    // DepthDS descriptor sets
    std::vector<VkDescriptorSetLayout> dsLayouts(MAX_FRAMES_IN_FLIGHT, depthDSDescriptorSetLayout);
    VkDescriptorSetAllocateInfo dsAllocInfo{};
    dsAllocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    dsAllocInfo.descriptorPool = descriptorPool;
    dsAllocInfo.descriptorSetCount = static_cast<uint32_t>(MAX_FRAMES_IN_FLIGHT);
    dsAllocInfo.pSetLayouts = dsLayouts.data();

    depthDSDescriptorSets.resize(MAX_FRAMES_IN_FLIGHT);
    if (vkAllocateDescriptorSets(device, &dsAllocInfo, depthDSDescriptorSets.data()) != VK_SUCCESS) {
        throw std::runtime_error("failed to allocate depthDS descriptor sets!");
    }

    for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
        VkDescriptorImageInfo depthInfo{};
        depthInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        depthInfo.imageView = depthTextureImageView;
        depthInfo.sampler = depthTextureSampler;

        VkDescriptorImageInfo albedoInfo{};
        albedoInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        albedoInfo.imageView = textureImageView;
        albedoInfo.sampler = textureSampler;

        VkDescriptorImageInfo normalInfo{};
        normalInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        normalInfo.imageView = normalTextureImageView;
        normalInfo.sampler = normalTextureSampler;

        VkDescriptorImageInfo newAlbedoInfo{};
        newAlbedoInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        newAlbedoInfo.imageView = albedoTextureImageView;
        newAlbedoInfo.sampler = albedoTextureSampler;

        std::vector<VkWriteDescriptorSet> writes(4);
        writes[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[0].dstSet = depthDSDescriptorSets[i];
        writes[0].dstBinding = 0;
        writes[0].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        writes[0].descriptorCount = 1;
        writes[0].pImageInfo = &depthInfo;

        writes[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[1].dstSet = depthDSDescriptorSets[i];
        writes[1].dstBinding = 1;
        writes[1].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        writes[1].descriptorCount = 1;
        writes[1].pImageInfo = &albedoInfo;

        writes[2].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[2].dstSet = depthDSDescriptorSets[i];
        writes[2].dstBinding = 2;
        writes[2].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        writes[2].descriptorCount = 1;
        writes[2].pImageInfo = &normalInfo;

        writes[3].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[3].dstSet = depthDSDescriptorSets[i];
        writes[3].dstBinding = 3;
        writes[3].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        writes[3].descriptorCount = 1;
        writes[3].pImageInfo = &newAlbedoInfo;

        vkUpdateDescriptorSets(device, static_cast<uint32_t>(writes.size()), writes.data(), 0, nullptr);
    }
    
    // Update RM descriptor sets to use depthDS output
    for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
        VkDescriptorImageInfo depthInfo{};
        depthInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        depthInfo.imageView = depthDSImageView;
        depthInfo.sampler = depthTextureSampler;

        VkWriteDescriptorSet write{};
        write.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        write.dstSet = descriptorSets[i];
        write.dstBinding = 1; // Replace original depth
        write.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        write.descriptorCount = 1;
        write.pImageInfo = &depthInfo;

        vkUpdateDescriptorSets(device, 1, &write, 0, nullptr);
    }

    // Final descriptor sets (for upscaling/presenting)
    std::vector<VkDescriptorSetLayout> finalLayouts(MAX_FRAMES_IN_FLIGHT, finalDescriptorSetLayout);
    VkDescriptorSetAllocateInfo finalAllocInfo{};
    finalAllocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    finalAllocInfo.descriptorPool = descriptorPool;
    finalAllocInfo.descriptorSetCount = static_cast<uint32_t>(MAX_FRAMES_IN_FLIGHT);
    finalAllocInfo.pSetLayouts = finalLayouts.data();

    finalDescriptorSets.resize(MAX_FRAMES_IN_FLIGHT);
    if (vkAllocateDescriptorSets(device, &finalAllocInfo, finalDescriptorSets.data()) != VK_SUCCESS) {
        throw std::runtime_error("failed to allocate final descriptor sets!");
    }

    for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
        // Initial binding (will be updated dynamically if needed)
        VkDescriptorImageInfo snrInfo{};
        snrInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        snrInfo.imageView = offscreenImageView; // DEBUG: Show RM (Offscreen) output
        snrInfo.sampler = offscreenSampler;

        VkDescriptorImageInfo colorInfo{};
        colorInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        colorInfo.imageView = textureImageView;
        colorInfo.sampler = textureSampler;

        VkWriteDescriptorSet writes[2]{};
        writes[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[0].dstSet = finalDescriptorSets[i];
        writes[0].dstBinding = 0;
        writes[0].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        writes[0].descriptorCount = 1;
        writes[0].pImageInfo = &snrInfo;

        writes[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[1].dstSet = finalDescriptorSets[i];
        writes[1].dstBinding = 1;
        writes[1].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        writes[1].descriptorCount = 1;
        writes[1].pImageInfo = &colorInfo;

        vkUpdateDescriptorSets(device, 2, writes, 0, nullptr);
    }
}

void VulkanRenderer::createCommandBuffers() {
    // Deprecated step if we record on the fly, but standard structure often pre-allocates them
    commandBuffers.resize(MAX_FRAMES_IN_FLIGHT);

    VkCommandBufferAllocateInfo allocInfo{};
    allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    allocInfo.commandPool = commandPool;
    allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    allocInfo.commandBufferCount = (uint32_t) commandBuffers.size();

    if (vkAllocateCommandBuffers(device, &allocInfo, commandBuffers.data()) != VK_SUCCESS) {
        throw std::runtime_error("failed to allocate command buffers!");
    }
}

// 17. Create Synchronization Objects.
// Vulkan is asynchronous. We need Semaphores (GPU-GPU sync) and Fences (CPU-GPU sync).
void VulkanRenderer::createSyncObjects() {
    imageAvailableSemaphores.resize(MAX_FRAMES_IN_FLIGHT);
    renderFinishedSemaphores.resize(MAX_FRAMES_IN_FLIGHT);
    inFlightFences.resize(MAX_FRAMES_IN_FLIGHT);

    VkSemaphoreCreateInfo semaphoreInfo{};
    semaphoreInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;

    VkFenceCreateInfo fenceInfo{};
    fenceInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    fenceInfo.flags = VK_FENCE_CREATE_SIGNALED_BIT; // Start signaled so we don't wait on the first frame

    for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
        if (vkCreateSemaphore(device, &semaphoreInfo, nullptr, &imageAvailableSemaphores[i]) != VK_SUCCESS ||
            vkCreateSemaphore(device, &semaphoreInfo, nullptr, &renderFinishedSemaphores[i]) != VK_SUCCESS ||
            vkCreateFence(device, &fenceInfo, nullptr, &inFlightFences[i]) != VK_SUCCESS) {
            throw std::runtime_error("failed to create synchronization objects for a frame!");
        }
    }
    
    // Also create command buffers here (needed for loop)
    createCommandBuffers();
}

// 18. Draw a Frame.
// This function orchestrates the rendering of a single frame:
// 1. Wait for the previous frame to finish.
// 2. Get the next available image from the swapchain.
// 3. Record commands (or use pre-recorded ones) to draw to that image.
// 4. Submit the commands to the GPU.
// 5. Present the image to the screen.
void VulkanRenderer::drawFrame() {
    // 1. Wait until the GPU has finished rendering the last frame.
    vkWaitForFences(device, 1, &inFlightFences[currentFrame], VK_TRUE, UINT64_MAX);

    // 2. Acquire an image from the swap chain
    uint32_t imageIndex;
    VkResult result = vkAcquireNextImageKHR(device, swapchain, UINT64_MAX, imageAvailableSemaphores[currentFrame], VK_NULL_HANDLE, &imageIndex);

    if (result == VK_ERROR_OUT_OF_DATE_KHR) {
        // The window has been resized and the swapchain is incompatible (not handled here for simplicity)
        return;
    } else if (result != VK_SUCCESS && result != VK_SUBOPTIMAL_KHR) {
        throw std::runtime_error("failed to acquire swap chain image!");
    }

    // Only reset the fence if we are submitting work
    vkResetFences(device, 1, &inFlightFences[currentFrame]);
    
    // Update texture logic for animation (CPU side)
    updateTexture();
    
    // Upload new texture data to the GPU immediately.
    // Note: In a production engine, this would use a separate transfer queue/command buffer to avoid stalling graphics.
    transitionImageLayout(textureImage, VK_FORMAT_R8G8B8A8_UNORM, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
    copyBufferToImage(stagingBuffer, textureImage, WIDTH, HEIGHT);
    transitionImageLayout(textureImage, VK_FORMAT_R8G8B8A8_UNORM, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);

    transitionImageLayout(depthTextureImage, VK_FORMAT_R8G8B8A8_UNORM, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
    copyBufferToImage(depthStagingBuffer, depthTextureImage, WIDTH, HEIGHT);
    transitionImageLayout(depthTextureImage, VK_FORMAT_R8G8B8A8_UNORM, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
    
    transitionImageLayout(normalTextureImage, VK_FORMAT_R8G8B8A8_UNORM, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
    copyBufferToImage(normalStagingBuffer, normalTextureImage, WIDTH, HEIGHT);
    transitionImageLayout(normalTextureImage, VK_FORMAT_R8G8B8A8_UNORM, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);

    transitionImageLayout(albedoTextureImage, VK_FORMAT_R8G8B8A8_UNORM, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
    copyBufferToImage(albedoStagingBuffer, albedoTextureImage, WIDTH, HEIGHT);
    transitionImageLayout(albedoTextureImage, VK_FORMAT_R8G8B8A8_UNORM, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);

    transitionImageLayout(mvTextureImage, VK_FORMAT_R8G8B8A8_UNORM, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
    copyBufferToImage(mvStagingBuffer, mvTextureImage, WIDTH, HEIGHT);
    transitionImageLayout(mvTextureImage, VK_FORMAT_R8G8B8A8_UNORM, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);

    // 3. Record drawing commands for this frame
    vkResetCommandBuffer(commandBuffers[currentFrame], 0);
    recordCommandBuffer(commandBuffers[currentFrame], imageIndex);

    // 4. Submit the command buffer
    VkSubmitInfo submitInfo{};
    submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;

    VkSemaphore waitSemaphores[] = {imageAvailableSemaphores[currentFrame]};
    VkPipelineStageFlags waitStages[] = {VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT};
    submitInfo.waitSemaphoreCount = 1;
    submitInfo.pWaitSemaphores = waitSemaphores; // Wait for image to be available
    submitInfo.pWaitDstStageMask = waitStages;
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &commandBuffers[currentFrame];

    VkSemaphore signalSemaphores[] = {renderFinishedSemaphores[currentFrame]};
    submitInfo.signalSemaphoreCount = 1;
    submitInfo.pSignalSemaphores = signalSemaphores; // Signal when rendering is finished

    if (vkQueueSubmit(graphicsQueue, 1, &submitInfo, inFlightFences[currentFrame]) != VK_SUCCESS) {
        throw std::runtime_error("failed to submit draw command buffer!");
    }

    // 5. Present the image (Show it on screen)
    VkPresentInfoKHR presentInfo{};
    presentInfo.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;
    presentInfo.waitSemaphoreCount = 1;
    presentInfo.pWaitSemaphores = signalSemaphores; // Wait for rendering to finish

    VkSwapchainKHR swapchains[] = {swapchain};
    presentInfo.swapchainCount = 1;
    presentInfo.pSwapchains = swapchains;
    presentInfo.pImageIndices = &imageIndex;

    vkQueuePresentKHR(presentQueue, &presentInfo);

    // Flip TNR history index (for temporal effects)
    tnrHistoryIndex = 1 - tnrHistoryIndex;

    // Advance to next frame index
    currentFrame = (currentFrame + 1) % MAX_FRAMES_IN_FLIGHT;
}

// 19. Record Commands.
// This function writes the actual GPU commands into the command buffer.
// It sets up the render passes, binds pipelines, descriptor sets, and issues draw calls.
void VulkanRenderer::recordCommandBuffer(VkCommandBuffer commandBuffer, uint32_t imageIndex) {
    VkCommandBufferBeginInfo beginInfo{};
    beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;

    if (vkBeginCommandBuffer(commandBuffer, &beginInfo) != VK_SUCCESS) {
        throw std::runtime_error("failed to begin recording command buffer!");
    }

    VkClearValue clearColor = {{{0.0f, 0.0f, 0.0f, 1.0f}}};

    // --- Pass 0: Depth Downsampling (DepthDS) ---
    // We render into the depthDSFramebuffer (Offscreen)
    VkRenderPassBeginInfo dsRenderPassInfo{};
    dsRenderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
    dsRenderPassInfo.renderPass = depthDSRenderPass;
    dsRenderPassInfo.framebuffer = depthDSFramebuffer;
    dsRenderPassInfo.renderArea.offset = {0, 0};
    dsRenderPassInfo.renderArea.extent = {RM_WIDTH, RM_HEIGHT};

    dsRenderPassInfo.clearValueCount = 1;
    dsRenderPassInfo.pClearValues = &clearColor;

    // Begin the pass
    vkCmdBeginRenderPass(commandBuffer, &dsRenderPassInfo, VK_SUBPASS_CONTENTS_INLINE);
    // Bind the pipeline (Depth Downsampling logic)
    vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, depthDSPipeline);

    // Set viewport/scissor dynamically
    VkViewport dsViewport{};
    dsViewport.x = 0.0f;
    dsViewport.y = 0.0f;
    dsViewport.width = (float)RM_WIDTH;
    dsViewport.height = (float)RM_HEIGHT;
    dsViewport.minDepth = 0.0f;
    dsViewport.maxDepth = 1.0f;
    vkCmdSetViewport(commandBuffer, 0, 1, &dsViewport);

    VkRect2D dsScissor{};
    dsScissor.offset = {0, 0};
    dsScissor.extent = {RM_WIDTH, RM_HEIGHT};
    vkCmdSetScissor(commandBuffer, 0, 1, &dsScissor);

    // Bind resources (Input images)
    vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, depthDSPipelineLayout, 0, 1, &depthDSDescriptorSets[currentFrame], 0, nullptr);
    // Draw a fullscreen quad (2 triangles = 6 vertices). The vertex shader generates the coordinates.
    vkCmdDraw(commandBuffer, 6, 1, 0, 0); 
    vkCmdEndRenderPass(commandBuffer);

    // --- Pass 1: Offscreen Ray Marching (RM) ---
    VkRenderPassBeginInfo offscreenRenderPassInfo{};
    offscreenRenderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
    offscreenRenderPassInfo.renderPass = offscreenRenderPass;
    offscreenRenderPassInfo.framebuffer = offscreenFramebuffer;
    offscreenRenderPassInfo.renderArea.offset = {0, 0};
    offscreenRenderPassInfo.renderArea.extent = {RM_WIDTH, RM_HEIGHT};

    offscreenRenderPassInfo.clearValueCount = 1;
    offscreenRenderPassInfo.pClearValues = &clearColor;

    vkCmdBeginRenderPass(commandBuffer, &offscreenRenderPassInfo, VK_SUBPASS_CONTENTS_INLINE);
    vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, offscreenPipeline);

    VkViewport rmViewport{};
    rmViewport.x = 0.0f;
    rmViewport.y = 0.0f;
    rmViewport.width = (float)RM_WIDTH;
    rmViewport.height = (float)RM_HEIGHT;
    rmViewport.minDepth = 0.0f;
    rmViewport.maxDepth = 1.0f;
    vkCmdSetViewport(commandBuffer, 0, 1, &rmViewport);

    VkRect2D rmScissor{};
    rmScissor.offset = {0, 0};
    rmScissor.extent = {RM_WIDTH, RM_HEIGHT};
    vkCmdSetScissor(commandBuffer, 0, 1, &rmScissor);

    vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, offscreenPipelineLayout, 0, 1, &descriptorSets[currentFrame], 0, nullptr);
    vkCmdDraw(commandBuffer, 6, 1, 0, 0);
    vkCmdEndRenderPass(commandBuffer);

    // --- Pass 2: Temporal Noise Reduction (TNR) ---
    VkRenderPassBeginInfo tnrRenderPassInfo{};
    tnrRenderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
    tnrRenderPassInfo.renderPass = tnrRenderPass;
    // Write to the NEXT history index, read from current history index in shader
    tnrRenderPassInfo.framebuffer = tnrFramebuffers[1 - tnrHistoryIndex];
    tnrRenderPassInfo.renderArea.offset = {0, 0};
    tnrRenderPassInfo.renderArea.extent = {RM_WIDTH, RM_HEIGHT};

    VkClearValue tnrClearValues[3] = {{{0.0f, 0.0f, 0.0f, 1.0f}}, {{0.0f, 0.0f, 0.0f, 1.0f}}, {{0.0f, 0.0f, 0.0f, 1.0f}}};
    tnrRenderPassInfo.clearValueCount = 3;
    tnrRenderPassInfo.pClearValues = tnrClearValues;

    vkCmdBeginRenderPass(commandBuffer, &tnrRenderPassInfo, VK_SUBPASS_CONTENTS_INLINE);
    vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, tnrPipeline);

    vkCmdSetViewport(commandBuffer, 0, 1, &rmViewport);
    vkCmdSetScissor(commandBuffer, 0, 1, &rmScissor);
    
    // TNR Logic uses a specific descriptor set to access history buffers
    // vkCmdBindDescriptorSets... (Assumed to be set up elsewhere or handled by layout/indices)
    // For brevity, assuming the bind happens correctly based on context or loop (not fully shown in snippet)

    vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, tnrPipelineLayout, 0, 1, &tnrDescriptorSets[currentFrame * 2 + tnrHistoryIndex], 0, nullptr);
    vkCmdDraw(commandBuffer, 6, 1, 0, 0);
    vkCmdEndRenderPass(commandBuffer);

    // --- Pass 3: SNR ---
    VkRenderPassBeginInfo snrRenderPassInfo{};
    snrRenderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
    snrRenderPassInfo.renderPass = snrRenderPass;
    snrRenderPassInfo.framebuffer = snrFramebuffers[1 - tnrHistoryIndex];
    snrRenderPassInfo.renderArea.offset = {0, 0};
    snrRenderPassInfo.renderArea.extent = {RM_WIDTH, RM_HEIGHT};

    snrRenderPassInfo.clearValueCount = 1;
    snrRenderPassInfo.pClearValues = &clearColor;

    vkCmdBeginRenderPass(commandBuffer, &snrRenderPassInfo, VK_SUBPASS_CONTENTS_INLINE);
    vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, snrPipeline);

    vkCmdSetViewport(commandBuffer, 0, 1, &rmViewport);
    vkCmdSetScissor(commandBuffer, 0, 1, &rmScissor);

    vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, snrPipelineLayout, 0, 1, &snrDescriptorSets[currentFrame], 0, nullptr);
    vkCmdDraw(commandBuffer, 6, 1, 0, 0);
    vkCmdEndRenderPass(commandBuffer);

    // --- Pass 3.5: SNR2 ---
    VkDescriptorImageInfo snrOutInfo{};
    snrOutInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    snrOutInfo.imageView = snrImageViews[1 - tnrHistoryIndex];
    snrOutInfo.sampler = offscreenSampler;

    VkWriteDescriptorSet snr2Write{};
    snr2Write.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    snr2Write.dstSet = snr2DescriptorSets[currentFrame];
    snr2Write.dstBinding = 0;
    snr2Write.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    snr2Write.descriptorCount = 1;
    snr2Write.pImageInfo = &snrOutInfo;
    vkUpdateDescriptorSets(device, 1, &snr2Write, 0, nullptr);

    VkRenderPassBeginInfo snr2RenderPassInfo{};
    snr2RenderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
    snr2RenderPassInfo.renderPass = snr2RenderPass;
    snr2RenderPassInfo.framebuffer = snr2Framebuffers[1 - tnrHistoryIndex];
    snr2RenderPassInfo.renderArea.offset = {0, 0};
    snr2RenderPassInfo.renderArea.extent = {RM_WIDTH, RM_HEIGHT};

    snr2RenderPassInfo.clearValueCount = 1;
    snr2RenderPassInfo.pClearValues = &clearColor;

    vkCmdBeginRenderPass(commandBuffer, &snr2RenderPassInfo, VK_SUBPASS_CONTENTS_INLINE);
    vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, snr2Pipeline);

    vkCmdSetViewport(commandBuffer, 0, 1, &rmViewport);
    vkCmdSetScissor(commandBuffer, 0, 1, &rmScissor);

    vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, snr2PipelineLayout, 0, 1, &snr2DescriptorSets[currentFrame], 0, nullptr);
    vkCmdDraw(commandBuffer, 6, 1, 0, 0);
    vkCmdEndRenderPass(commandBuffer);

    // --- Pass 3.6: Compute Fresnel ---
    VkRenderPassBeginInfo fresnelPassInfo{};
    fresnelPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
    fresnelPassInfo.renderPass = computeFresnelRenderPass;
    fresnelPassInfo.framebuffer = computeFresnelFramebuffer;
    fresnelPassInfo.renderArea.offset = {0, 0};
    fresnelPassInfo.renderArea.extent = {RM_WIDTH, RM_HEIGHT};
    fresnelPassInfo.clearValueCount = 1;
    fresnelPassInfo.pClearValues = &clearColor;

    vkCmdBeginRenderPass(commandBuffer, &fresnelPassInfo, VK_SUBPASS_CONTENTS_INLINE);
    vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, computeFresnelPipeline);
    vkCmdSetViewport(commandBuffer, 0, 1, &rmViewport);
    vkCmdSetScissor(commandBuffer, 0, 1, &rmScissor);
    vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, computeFresnelPipelineLayout, 0, 1, &computeFresnelDescriptorSets[currentFrame], 0, nullptr);
    vkCmdDraw(commandBuffer, 6, 1, 0, 0);
    vkCmdEndRenderPass(commandBuffer);

    // --- Pass 3.7: TNR2 ---
    VkRenderPassBeginInfo tnr2RenderPassInfo{};
    tnr2RenderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
    tnr2RenderPassInfo.renderPass = tnr2RenderPass;
    tnr2RenderPassInfo.framebuffer = tnr2Framebuffers[1 - tnrHistoryIndex];
    tnr2RenderPassInfo.renderArea.offset = {0, 0};
    tnr2RenderPassInfo.renderArea.extent = {RM_WIDTH, RM_HEIGHT};
    tnr2RenderPassInfo.clearValueCount = 2; // Color, Info
    VkClearValue tnr2ClearValues[2] = {clearColor, clearColor};
    tnr2RenderPassInfo.pClearValues = tnr2ClearValues;

    vkCmdBeginRenderPass(commandBuffer, &tnr2RenderPassInfo, VK_SUBPASS_CONTENTS_INLINE);
    vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, tnr2Pipeline);
    vkCmdSetViewport(commandBuffer, 0, 1, &rmViewport);
    vkCmdSetScissor(commandBuffer, 0, 1, &rmScissor);
    vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, tnr2PipelineLayout, 0, 1, &tnr2DescriptorSets[currentFrame * 2 + tnrHistoryIndex], 0, nullptr);
    vkCmdDraw(commandBuffer, 6, 1, 0, 0);
    vkCmdEndRenderPass(commandBuffer);

    // --- Pass 4: Final Upscale ---
    VkRenderPassBeginInfo renderPassInfo{};
    renderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
    renderPassInfo.renderPass = renderPass;
    renderPassInfo.framebuffer = swapchainFramebuffers[imageIndex];
    renderPassInfo.renderArea.offset = {0, 0};
    renderPassInfo.renderArea.extent = swapchainExtent;

    renderPassInfo.clearValueCount = 1;
    renderPassInfo.pClearValues = &clearColor;

    vkCmdBeginRenderPass(commandBuffer, &renderPassInfo, VK_SUBPASS_CONTENTS_INLINE);
    vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, finalPipeline);

    VkViewport finalViewport{};
    finalViewport.x = 0.0f;
    finalViewport.y = 0.0f;
    finalViewport.width = (float)swapchainExtent.width;
    finalViewport.height = (float)swapchainExtent.height;
    finalViewport.minDepth = 0.0f;
    finalViewport.maxDepth = 1.0f;
    vkCmdSetViewport(commandBuffer, 0, 1, &finalViewport);

    VkRect2D finalScissor{};
    finalScissor.offset = {0, 0};
    finalScissor.extent = swapchainExtent;
    vkCmdSetScissor(commandBuffer, 0, 1, &finalScissor);

    // Update final descriptor set to read from the TNR2 output
    VkDescriptorImageInfo resultInfo{};
    resultInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    resultInfo.imageView = tnr2ImageViews[1 - tnrHistoryIndex]; // TNR2_out0
    resultInfo.sampler = offscreenSampler;

    VkWriteDescriptorSet resultWrite{};
    resultWrite.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    resultWrite.dstSet = finalDescriptorSets[currentFrame];
    resultWrite.dstBinding = 0;
    resultWrite.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    resultWrite.descriptorCount = 1;
    resultWrite.pImageInfo = &resultInfo;

    // Update to show TNR2 Output
    vkUpdateDescriptorSets(device, 1, &resultWrite, 0, nullptr);

    vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, finalPipelineLayout, 0, 1, &finalDescriptorSets[currentFrame], 0, nullptr);
    vkCmdDraw(commandBuffer, 6, 1, 0, 0);
    vkCmdEndRenderPass(commandBuffer);

    if (vkEndCommandBuffer(commandBuffer) != VK_SUCCESS) {
        throw std::runtime_error("failed to record command buffer!");
    }
}

void VulkanRenderer::updateTexture() {
    frameDelayCounter++;
    if (frameDelayCounter < frameDelay) {
        return;
    }
    frameDelayCounter = 0;
    
    // Generate filenames
    std::ostringstream oss, doss, noss;
    oss << COLOR_PATH_PREFIX << std::setw(4) << std::setfill('0') << currentFrameIndex << FILE_EXTENSION;
    doss << DEPTH_PATH_PREFIX << std::setw(4) << std::setfill('0') << currentFrameIndex << FILE_EXTENSION;
    noss << NORMAL_PATH_PREFIX << std::setw(4) << std::setfill('0') << currentFrameIndex << FILE_EXTENSION;
    
    std::ostringstream aoss;
    aoss << ALBEDO_PATH_PREFIX << std::setw(4) << std::setfill('0') << currentFrameIndex << FILE_EXTENSION;

    void* data;
    vkMapMemory(device, stagingBufferMemory, 0, WIDTH * HEIGHT * 4, 0, &data);
    loadRawImage(oss.str(), data, COLOR_PATH_PREFIX);
    vkUnmapMemory(device, stagingBufferMemory);

    void* ddata;
    vkMapMemory(device, depthStagingBufferMemory, 0, WIDTH * HEIGHT * 4, 0, &ddata);
    loadRawImage(doss.str(), ddata, DEPTH_PATH_PREFIX);
    vkUnmapMemory(device, depthStagingBufferMemory);

    void* ndata;
    vkMapMemory(device, normalStagingBufferMemory, 0, WIDTH * HEIGHT * 4, 0, &ndata);
    loadRawImage(noss.str(), ndata, NORMAL_PATH_PREFIX);
    vkUnmapMemory(device, normalStagingBufferMemory);

    void* adata;
    vkMapMemory(device, albedoStagingBufferMemory, 0, WIDTH * HEIGHT * 4, 0, &adata);
    loadRawImage(aoss.str(), adata, ALBEDO_PATH_PREFIX);
    vkUnmapMemory(device, albedoStagingBufferMemory);
    
    std::ostringstream mvoss;
    mvoss << MV_PATH_PREFIX << std::setw(4) << std::setfill('0') << currentFrameIndex << FILE_EXTENSION;
    void* mvdata;
    vkMapMemory(device, mvStagingBufferMemory, 0, WIDTH * HEIGHT * 4, 0, &mvdata);
    loadRawImage(mvoss.str(), mvdata, MV_PATH_PREFIX);
    vkUnmapMemory(device, mvStagingBufferMemory);

    currentFrameIndex++;
    if (currentFrameIndex >= 148) {
        currentFrameIndex = 0;
    }
}

void VulkanRenderer::loadRawImage(const std::string& filename, void* pixels, const std::string& fallbackPrefix) {
    std::ifstream file(filename, std::ios::ate | std::ios::binary);
    
    size_t expectedSize = WIDTH * HEIGHT * 4;
    
    if (!file.is_open()) {
        // Try fallback if running from build directory
        file.open("../" + filename, std::ios::ate | std::ios::binary);
    }

    if (!file.is_open()) {
        std::cerr << "Error: Could not open " << filename << ". Check if working directory is correct." << std::endl;
        
        // Check if we should loop back to 0 using the provided prefix
        if (!fallbackPrefix.empty() && currentFrameIndex > 0) {
            std::ostringstream oss;
            oss << fallbackPrefix << std::setw(4) << std::setfill('0') << 0 << FILE_EXTENSION;
            std::string restartPath = oss.str();
            std::ifstream restartFile(restartPath, std::ios::ate | std::ios::binary);
            if (!restartFile.is_open()) {
                restartFile.open("../" + restartPath, std::ios::ate | std::ios::binary);
            }
            if (restartFile.is_open()) {
               if (restartFile.tellg() == expectedSize) {
                   restartFile.seekg(0);
                   restartFile.read((char*)pixels, expectedSize);
                   restartFile.close();
                   goto post_load_flip;
               }
            } 
        }
        
        // If still nothing, fill with 0 (Black for color, 0.0f for depth)
        std::memset(pixels, 0, expectedSize);
        return;
    }
    
    {
        size_t fileSize = (size_t) file.tellg();
        if (fileSize != expectedSize) {
            std::cerr << "Warning: Incorrect file size for " << filename << std::endl;
            uint32_t* pDiv = (uint32_t*)pixels;
            for (size_t i=0; i < WIDTH * HEIGHT; i++) {
                 pDiv[i] = 0xFF00FF00; // Green warning
            }
            file.close();
            return;
        }
        
        file.seekg(0);
        file.read((char*)pixels, fileSize);
        file.close();
    }

post_load_flip:
    // Flip vertically in-place
    size_t rowSize = WIDTH * 4;
    std::vector<char> rowBuffer(rowSize);
    char* data = (char*)pixels;
    for (size_t y = 0; y < HEIGHT / 2; y++) {
        char* rowTop = data + (y * rowSize);
        char* rowBottom = data + ((HEIGHT - 1 - y) * rowSize);
        std::memcpy(rowBuffer.data(), rowTop, rowSize);
        std::memcpy(rowTop, rowBottom, rowSize);
        std::memcpy(rowBottom, rowBuffer.data(), rowSize);
    }
}


// Helpers
bool VulkanRenderer::checkValidationLayerSupport() {
    uint32_t layerCount;
    vkEnumerateInstanceLayerProperties(&layerCount, nullptr);

    std::vector<VkLayerProperties> availableLayers(layerCount);
    vkEnumerateInstanceLayerProperties(&layerCount, availableLayers.data());

    for (const char* layerName : validationLayers) {
        bool layerFound = false;
        for (const auto& layerProperties : availableLayers) {
            if (strcmp(layerName, layerProperties.layerName) == 0) {
                layerFound = true;
                break;
            }
        }
        if (!layerFound) {
            return false;
        }
    }
    return true;
}

std::vector<const char*> VulkanRenderer::getRequiredExtensions() {
    uint32_t glfwExtensionCount = 0;
    const char** glfwExtensions;
    glfwExtensions = glfwGetRequiredInstanceExtensions(&glfwExtensionCount);

    std::vector<const char*> extensions(glfwExtensions, glfwExtensions + glfwExtensionCount);

    if (enableValidationLayers) {
        extensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
    }
    
    // Check for portability enumeration support
    uint32_t extensionCount = 0;
    vkEnumerateInstanceExtensionProperties(nullptr, &extensionCount, nullptr);
    std::vector<VkExtensionProperties> availableExtensions(extensionCount);
    vkEnumerateInstanceExtensionProperties(nullptr, &extensionCount, availableExtensions.data());

    for (const auto& extension : availableExtensions) {
        if (strcmp(extension.extensionName, VK_KHR_PORTABILITY_ENUMERATION_EXTENSION_NAME) == 0) {
            extensions.push_back(VK_KHR_PORTABILITY_ENUMERATION_EXTENSION_NAME);
        }
        if (strcmp(extension.extensionName, VK_KHR_GET_PHYSICAL_DEVICE_PROPERTIES_2_EXTENSION_NAME) == 0) {
            extensions.push_back(VK_KHR_GET_PHYSICAL_DEVICE_PROPERTIES_2_EXTENSION_NAME);
        }
    }
    
    return extensions;
}

// 20. Helper: Read File.
// Reads a binary file (like a shader) from disk into a byte vector.
std::vector<char> VulkanRenderer::readFile(const std::string& filename) {
    std::ifstream file(filename, std::ios::ate | std::ios::binary);

    if (!file.is_open()) {
        throw std::runtime_error("failed to open file: " + filename);
    }

    size_t fileSize = (size_t) file.tellg();
    std::vector<char> buffer(fileSize);

    file.seekg(0);
    file.read(buffer.data(), fileSize);
    file.close();

    return buffer;
}


VKAPI_ATTR VkBool32 VKAPI_CALL VulkanRenderer::debugCallback(VkDebugUtilsMessageSeverityFlagBitsEXT messageSeverity, VkDebugUtilsMessageTypeFlagsEXT messageType, const VkDebugUtilsMessengerCallbackDataEXT* pCallbackData, void* pUserData) {
    if (messageSeverity >= VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT) {
        std::cerr << "validation layer: " << pCallbackData->pMessage << std::endl;
    }
    return VK_FALSE;
}

// Boilerplate Helpers
// Helper: Find Memory Type.
// Vulkan requires us to manually find the right type of memory on the GPU (e.g., VRAM vs System RAM).
uint32_t VulkanRenderer::findMemoryType(uint32_t typeFilter, VkMemoryPropertyFlags properties) {
    VkPhysicalDeviceMemoryProperties memProperties;
    vkGetPhysicalDeviceMemoryProperties(physicalDevice, &memProperties);

    for (uint32_t i = 0; i < memProperties.memoryTypeCount; i++) {
        // Check if the bitmask asks for this type AND if it has the required properties (like being Host Visible)
        if ((typeFilter & (1 << i)) && (memProperties.memoryTypes[i].propertyFlags & properties) == properties) {
            return i;
        }
    }
    throw std::runtime_error("failed to find suitable memory type!");
}

// Helper: Create Buffer.
// Allocates a buffer (for vertices, indices, or staging) on the GPU.
void VulkanRenderer::createBuffer(VkDeviceSize size, VkBufferUsageFlags usage, VkMemoryPropertyFlags properties, VkBuffer& buffer, VkDeviceMemory& bufferMemory) {
    VkBufferCreateInfo bufferInfo{};
    bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bufferInfo.size = size;
    bufferInfo.usage = usage;
    bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE; // Only used by graphics queue

    if (vkCreateBuffer(device, &bufferInfo, nullptr, &buffer) != VK_SUCCESS) {
        throw std::runtime_error("failed to create buffer!");
    }

    // Get memory requirements (alignment, size)
    VkMemoryRequirements memRequirements;
    vkGetBufferMemoryRequirements(device, buffer, &memRequirements);

    VkMemoryAllocateInfo allocInfo{};
    allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    allocInfo.allocationSize = memRequirements.size;
    allocInfo.memoryTypeIndex = findMemoryType(memRequirements.memoryTypeBits, properties);

    // Allocate the actual memory
    if (vkAllocateMemory(device, &allocInfo, nullptr, &bufferMemory) != VK_SUCCESS) {
        throw std::runtime_error("failed to allocate buffer memory!");
    }
    // Bind the memory to the buffer handle
    vkBindBufferMemory(device, buffer, bufferMemory, 0);
}

// Helper: Create Image.
// Allocates an image (texture, depth attachment, etc.) on the GPU.
void VulkanRenderer::createImage(uint32_t width, uint32_t height, VkFormat format, VkImageTiling tiling, VkImageUsageFlags usage, VkMemoryPropertyFlags properties, VkImage& image, VkDeviceMemory& imageMemory) {
    VkImageCreateInfo imageInfo{};
    imageInfo.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    imageInfo.imageType = VK_IMAGE_TYPE_2D;
    imageInfo.extent.width = width;
    imageInfo.extent.height = height;
    imageInfo.extent.depth = 1;
    imageInfo.mipLevels = 1;
    imageInfo.arrayLayers = 1;
    imageInfo.format = format;
    imageInfo.tiling = tiling;
    imageInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    imageInfo.usage = usage;
    imageInfo.samples = VK_SAMPLE_COUNT_1_BIT;
    imageInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

    if (vkCreateImage(device, &imageInfo, nullptr, &image) != VK_SUCCESS) {
        throw std::runtime_error("failed to create image!");
    }

    VkMemoryRequirements memRequirements;
    vkGetImageMemoryRequirements(device, image, &memRequirements);

    VkMemoryAllocateInfo allocInfo{};
    allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    allocInfo.allocationSize = memRequirements.size;
    allocInfo.memoryTypeIndex = findMemoryType(memRequirements.memoryTypeBits, properties);

    if (vkAllocateMemory(device, &allocInfo, nullptr, &imageMemory) != VK_SUCCESS) {
        throw std::runtime_error("failed to allocate image memory!");
    }

    vkBindImageMemory(device, image, imageMemory, 0);
}


// Helper: Create Image View.
// Creates a view into an image, specifying how to interpret it (color, depth, etc.).
VkImageView VulkanRenderer::createImageView(VkImage image, VkFormat format) {
    VkImageViewCreateInfo viewInfo{};
    viewInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    viewInfo.image = image;
    viewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
    viewInfo.format = format;
    viewInfo.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    viewInfo.subresourceRange.baseMipLevel = 0;
    viewInfo.subresourceRange.levelCount = 1;
    viewInfo.subresourceRange.baseArrayLayer = 0;
    viewInfo.subresourceRange.layerCount = 1;

    VkImageView imageView;
    if (vkCreateImageView(device, &viewInfo, nullptr, &imageView) != VK_SUCCESS) {
        throw std::runtime_error("failed to create texture image view!");
    }
    return imageView;
}


// Helper: Transition Image Layout.
// Uses a "Pipeline Barrier" to transition an image from one layout to another (e.g., Undefined -> Transfer Dest).
// This ensures the GPU has finished unrelated work and flushes caches as needed.
void VulkanRenderer::transitionImageLayout(VkImage image, VkFormat format, VkImageLayout oldLayout, VkImageLayout newLayout) {
    // Allocation of a temporary command buffer for the barrier command
    VkCommandBufferAllocateInfo allocInfo{};
    allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    allocInfo.commandPool = commandPool;
    allocInfo.commandBufferCount = 1;

    VkCommandBuffer commandBuffer;
    vkAllocateCommandBuffers(device, &allocInfo, &commandBuffer);

    VkCommandBufferBeginInfo beginInfo{};
    beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;

    vkBeginCommandBuffer(commandBuffer, &beginInfo);

    VkImageMemoryBarrier barrier{};
    barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    barrier.oldLayout = oldLayout;
    barrier.newLayout = newLayout;
    barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.image = image;
    barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    barrier.subresourceRange.baseMipLevel = 0;
    barrier.subresourceRange.levelCount = 1;
    barrier.subresourceRange.baseArrayLayer = 0;
    barrier.subresourceRange.layerCount = 1;

    VkPipelineStageFlags sourceStage;
    VkPipelineStageFlags destinationStage;

    // Define the source and destination access masks and stages based on the transition
    if (oldLayout == VK_IMAGE_LAYOUT_UNDEFINED && newLayout == VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL) {
        barrier.srcAccessMask = 0;
        barrier.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        sourceStage = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
        destinationStage = VK_PIPELINE_STAGE_TRANSFER_BIT;
    } else if (oldLayout == VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL && newLayout == VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL) {
        barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
        sourceStage = VK_PIPELINE_STAGE_TRANSFER_BIT;
        destinationStage = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
    } else if (oldLayout == VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL && newLayout == VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL) {
        barrier.srcAccessMask = VK_ACCESS_SHADER_READ_BIT;
        barrier.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        sourceStage = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
        destinationStage = VK_PIPELINE_STAGE_TRANSFER_BIT;
    } else if (oldLayout == VK_IMAGE_LAYOUT_UNDEFINED && newLayout == VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL) {
        barrier.srcAccessMask = 0;
        barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
        sourceStage = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
        destinationStage = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
    } else {
        throw std::invalid_argument("unsupported layout transition!");
    }

    vkCmdPipelineBarrier(commandBuffer, sourceStage, destinationStage, 0, 0, nullptr, 0, nullptr, 1, &barrier);

    vkEndCommandBuffer(commandBuffer);

    // Submit and wait for completion (Not efficient for every single transition, but simple for initialization)
    VkSubmitInfo submitInfo{};
    submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &commandBuffer;

    vkQueueSubmit(graphicsQueue, 1, &submitInfo, VK_NULL_HANDLE);
    vkQueueWaitIdle(graphicsQueue);

    vkFreeCommandBuffers(device, commandPool, 1, &commandBuffer);
}

// Helper: Copy Buffer To Image.
// Copies data from a CPU-visible buffer (staging) to a GPU image.
void VulkanRenderer::copyBufferToImage(VkBuffer buffer, VkImage image, uint32_t width, uint32_t height) {
    VkCommandBufferAllocateInfo allocInfo{};
    allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    allocInfo.commandPool = commandPool;
    allocInfo.commandBufferCount = 1;

    VkCommandBuffer commandBuffer;
    vkAllocateCommandBuffers(device, &allocInfo, &commandBuffer);

    VkCommandBufferBeginInfo beginInfo{};
    beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;

    vkBeginCommandBuffer(commandBuffer, &beginInfo);

    VkBufferImageCopy region{};
    region.bufferOffset = 0;
    region.bufferRowLength = 0;
    region.bufferImageHeight = 0;
    region.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    region.imageSubresource.mipLevel = 0;
    region.imageSubresource.baseArrayLayer = 0;
    region.imageSubresource.layerCount = 1;
    region.imageOffset = {0, 0, 0};
    region.imageExtent = {width, height, 1};

    vkCmdCopyBufferToImage(commandBuffer, buffer, image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region);

    vkEndCommandBuffer(commandBuffer);

    VkSubmitInfo submitInfo{};
    submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &commandBuffer;

    vkQueueSubmit(graphicsQueue, 1, &submitInfo, VK_NULL_HANDLE);
    vkQueueWaitIdle(graphicsQueue);

    vkFreeCommandBuffers(device, commandPool, 1, &commandBuffer);
}

// Helper: Create Shader Module.
// Wraps the SPIR-V bytecode into a Vulkan Shader Module object.
VkShaderModule VulkanRenderer::createShaderModule(const std::vector<char>& code) {
    VkShaderModuleCreateInfo createInfo{};
    createInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    createInfo.codeSize = code.size();
    createInfo.pCode = reinterpret_cast<const uint32_t*>(code.data());

    VkShaderModule shaderModule;
    if (vkCreateShaderModule(device, &createInfo, nullptr, &shaderModule) != VK_SUCCESS) {
        throw std::runtime_error("failed to create shader module!");
    }

    return shaderModule;
}

void VulkanRenderer::createNormalTextureImage() {
    VkDeviceSize imageSize = WIDTH * HEIGHT * 4;
    createBuffer(imageSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, normalStagingBuffer, normalStagingBufferMemory);
    createImage(WIDTH, HEIGHT, VK_FORMAT_R8G8B8A8_UNORM, VK_IMAGE_TILING_OPTIMAL, VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, normalTextureImage, normalTextureImageMemory);
    transitionImageLayout(normalTextureImage, VK_FORMAT_R8G8B8A8_UNORM, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
    transitionImageLayout(normalTextureImage, VK_FORMAT_R8G8B8A8_UNORM, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
}

void VulkanRenderer::createNormalTextureImageView() {
    normalTextureImageView = createImageView(normalTextureImage, VK_FORMAT_R8G8B8A8_UNORM);
}

void VulkanRenderer::createNormalTextureSampler() {
    VkSamplerCreateInfo samplerInfo{};
    samplerInfo.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
    samplerInfo.magFilter = VK_FILTER_LINEAR;
    samplerInfo.minFilter = VK_FILTER_LINEAR;
    samplerInfo.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    samplerInfo.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    samplerInfo.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    samplerInfo.anisotropyEnable = VK_FALSE;
    samplerInfo.borderColor = VK_BORDER_COLOR_INT_OPAQUE_BLACK;
    samplerInfo.unnormalizedCoordinates = VK_FALSE;
    samplerInfo.compareEnable = VK_FALSE;
    samplerInfo.compareOp = VK_COMPARE_OP_ALWAYS;
    samplerInfo.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;

    if (vkCreateSampler(device, &samplerInfo, nullptr, &normalTextureSampler) != VK_SUCCESS) {
        throw std::runtime_error("failed to create normal texture sampler!");
    }
}

void VulkanRenderer::createDepthDSResources() {
    createImage(RM_WIDTH, RM_HEIGHT, VK_FORMAT_R16G16B16A16_SFLOAT, VK_IMAGE_TILING_OPTIMAL, VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, depthDSImage, depthDSImageMemory);
    depthDSImageView = createImageView(depthDSImage, VK_FORMAT_R16G16B16A16_SFLOAT);
    transitionImageLayout(depthDSImage, VK_FORMAT_R16G16B16A16_SFLOAT, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);

    // Framebuffer
    VkFramebufferCreateInfo framebufferInfo{};
    framebufferInfo.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
    framebufferInfo.renderPass = depthDSRenderPass;
    framebufferInfo.attachmentCount = 1;
    framebufferInfo.pAttachments = &depthDSImageView;
    framebufferInfo.width = RM_WIDTH;
    framebufferInfo.height = RM_HEIGHT;
    framebufferInfo.layers = 1;

    if (vkCreateFramebuffer(device, &framebufferInfo, nullptr, &depthDSFramebuffer) != VK_SUCCESS) {
        throw std::runtime_error("failed to create depthDS framebuffer!");
    }
}

void VulkanRenderer::createMVTextureImage() {
    VkDeviceSize imageSize = WIDTH * HEIGHT * 4;
    createBuffer(imageSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, mvStagingBuffer, mvStagingBufferMemory);
    createImage(WIDTH, HEIGHT, VK_FORMAT_R8G8B8A8_UNORM, VK_IMAGE_TILING_OPTIMAL, VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, mvTextureImage, mvTextureImageMemory);
    transitionImageLayout(mvTextureImage, VK_FORMAT_R8G8B8A8_UNORM, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
    transitionImageLayout(mvTextureImage, VK_FORMAT_R8G8B8A8_UNORM, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
}

void VulkanRenderer::createMVTextureImageView() {
    mvTextureImageView = createImageView(mvTextureImage, VK_FORMAT_R8G8B8A8_UNORM);
}

void VulkanRenderer::createMVTextureSampler() {
    VkSamplerCreateInfo samplerInfo{};
    samplerInfo.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
    samplerInfo.magFilter = VK_FILTER_LINEAR;
    samplerInfo.minFilter = VK_FILTER_LINEAR;
    samplerInfo.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    samplerInfo.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    samplerInfo.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    samplerInfo.anisotropyEnable = VK_FALSE;
    samplerInfo.borderColor = VK_BORDER_COLOR_INT_OPAQUE_BLACK;
    samplerInfo.unnormalizedCoordinates = VK_FALSE;
    samplerInfo.compareEnable = VK_FALSE;
    samplerInfo.compareOp = VK_COMPARE_OP_ALWAYS;
    samplerInfo.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;

    if (vkCreateSampler(device, &samplerInfo, nullptr, &mvTextureSampler) != VK_SUCCESS) {
        throw std::runtime_error("failed to create MV texture sampler!");
    }
}

void VulkanRenderer::createAlbedoTextureImage() {
    VkDeviceSize imageSize = WIDTH * HEIGHT * 4;
    createBuffer(imageSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, albedoStagingBuffer, albedoStagingBufferMemory);
    createImage(WIDTH, HEIGHT, VK_FORMAT_R8G8B8A8_UNORM, VK_IMAGE_TILING_OPTIMAL, VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, albedoTextureImage, albedoTextureImageMemory);
    transitionImageLayout(albedoTextureImage, VK_FORMAT_R8G8B8A8_UNORM, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
    transitionImageLayout(albedoTextureImage, VK_FORMAT_R8G8B8A8_UNORM, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
}

void VulkanRenderer::createAlbedoTextureImageView() {
    albedoTextureImageView = createImageView(albedoTextureImage, VK_FORMAT_R8G8B8A8_UNORM);
}

void VulkanRenderer::createAlbedoTextureSampler() {
    VkSamplerCreateInfo samplerInfo{};
    samplerInfo.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
    samplerInfo.magFilter = VK_FILTER_LINEAR;
    samplerInfo.minFilter = VK_FILTER_LINEAR;
    samplerInfo.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    samplerInfo.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    samplerInfo.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    samplerInfo.anisotropyEnable = VK_FALSE;
    samplerInfo.borderColor = VK_BORDER_COLOR_INT_OPAQUE_BLACK;
    samplerInfo.unnormalizedCoordinates = VK_FALSE;
    samplerInfo.compareEnable = VK_FALSE;
    samplerInfo.compareOp = VK_COMPARE_OP_ALWAYS;
    samplerInfo.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;

    if (vkCreateSampler(device, &samplerInfo, nullptr, &albedoTextureSampler) != VK_SUCCESS) {
        throw std::runtime_error("failed to create Albedo texture sampler!");
    }
}

void VulkanRenderer::createTNRResources() {
    // Intermediate output image
    createImage(RM_WIDTH, RM_HEIGHT, VK_FORMAT_R16G16B16A16_SFLOAT, VK_IMAGE_TILING_OPTIMAL, VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, tnrIntermediateColorImage, tnrIntermediateColorImageMemory);
    tnrIntermediateColorImageView = createImageView(tnrIntermediateColorImage, VK_FORMAT_R16G16B16A16_SFLOAT);
    transitionImageLayout(tnrIntermediateColorImage, VK_FORMAT_R16G16B16A16_SFLOAT, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);

    // Out2 Image
    createImage(RM_WIDTH, RM_HEIGHT, VK_FORMAT_R16G16B16A16_SFLOAT, VK_IMAGE_TILING_OPTIMAL, VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, tnrOut2Image, tnrOut2ImageMemory);
    tnrOut2ImageView = createImageView(tnrOut2Image, VK_FORMAT_R16G16B16A16_SFLOAT);
    transitionImageLayout(tnrOut2Image, VK_FORMAT_R16G16B16A16_SFLOAT, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);

    // 1. Render Pass
    VkAttachmentDescription colorAttachment{};
    colorAttachment.format = VK_FORMAT_R16G16B16A16_SFLOAT;
    colorAttachment.samples = VK_SAMPLE_COUNT_1_BIT;
    colorAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
    colorAttachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    colorAttachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    colorAttachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    colorAttachment.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    colorAttachment.finalLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

    VkAttachmentDescription infoAttachment{};
    infoAttachment.format = VK_FORMAT_R16G16B16A16_SFLOAT;
    infoAttachment.samples = VK_SAMPLE_COUNT_1_BIT;
    infoAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
    infoAttachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    infoAttachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    infoAttachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    infoAttachment.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    infoAttachment.finalLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

    VkAttachmentDescription out2Attachment{};
    out2Attachment.format = VK_FORMAT_R16G16B16A16_SFLOAT;
    out2Attachment.samples = VK_SAMPLE_COUNT_1_BIT;
    out2Attachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
    out2Attachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    out2Attachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    out2Attachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    out2Attachment.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    out2Attachment.finalLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

    VkAttachmentReference colorReference = {0, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL};
    VkAttachmentReference infoReference = {1, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL};
    VkAttachmentReference out2Reference = {2, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL};
    VkAttachmentReference attachmentsRef[] = {colorReference, infoReference, out2Reference};

    VkSubpassDescription subpass{};
    subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
    subpass.colorAttachmentCount = 3;
    subpass.pColorAttachments = attachmentsRef;

    VkAttachmentDescription attachments[] = {colorAttachment, infoAttachment, out2Attachment};
    VkRenderPassCreateInfo renderPassInfo{};
    renderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
    renderPassInfo.attachmentCount = 3;
    renderPassInfo.pAttachments = attachments;
    renderPassInfo.subpassCount = 1;
    renderPassInfo.pSubpasses = &subpass;

    if (vkCreateRenderPass(device, &renderPassInfo, nullptr, &tnrRenderPass) != VK_SUCCESS) {
        throw std::runtime_error("failed to create TNR render pass!");
    }

    // 2. Info Images (Double buffered for flip)
    for (int i = 0; i < 2; i++) {
        createImage(RM_WIDTH, RM_HEIGHT, VK_FORMAT_R16G16B16A16_SFLOAT, VK_IMAGE_TILING_OPTIMAL, VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, tnrInfoImages[i], tnrInfoImageMemories[i]);
        tnrInfoImageViews[i] = createImageView(tnrInfoImages[i], VK_FORMAT_R16G16B16A16_SFLOAT);
        transitionImageLayout(tnrInfoImages[i], VK_FORMAT_R16G16B16A16_SFLOAT, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);

        VkImageView attachmentsFB[] = {tnrIntermediateColorImageView, tnrInfoImageViews[i], tnrOut2ImageView};
        VkFramebufferCreateInfo framebufferInfo{};
        framebufferInfo.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
        framebufferInfo.renderPass = tnrRenderPass;
        framebufferInfo.attachmentCount = 3;
        framebufferInfo.pAttachments = attachmentsFB;
        framebufferInfo.width = RM_WIDTH;
        framebufferInfo.height = RM_HEIGHT;
        framebufferInfo.layers = 1;

        if (vkCreateFramebuffer(device, &framebufferInfo, nullptr, &tnrFramebuffers[i]) != VK_SUCCESS) {
            throw std::runtime_error("failed to create TNR framebuffer!");
        }
    }

    // 3. Descriptor Set Layout
    VkDescriptorSetLayoutBinding bindings[6]{};
    for(int i=0; i<6; i++) {
        bindings[i].binding = i;
        bindings[i].descriptorCount = 1;
        bindings[i].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        bindings[i].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
    }

    VkDescriptorSetLayoutCreateInfo layoutInfo{};
    layoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    layoutInfo.bindingCount = 6;
    layoutInfo.pBindings = bindings;

    if (vkCreateDescriptorSetLayout(device, &layoutInfo, nullptr, &tnrDescriptorSetLayout) != VK_SUCCESS) {
        throw std::runtime_error("failed to create TNR descriptor set layout!");
    }

    // 4. Pipeline
    auto tnrFragCode = readFile(std::string(SHADER_DIR) + "/TNR.frag.spv");
    VkShaderModule tnrFragModule = createShaderModule(tnrFragCode);
    
    auto vertShaderCode = readFile(std::string(SHADER_DIR) + "/shader.vert.spv");
    VkShaderModule vertShaderModule = createShaderModule(vertShaderCode);

    VkPipelineShaderStageCreateInfo stages[2]{};
    stages[0].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    stages[0].stage = VK_SHADER_STAGE_VERTEX_BIT;
    stages[0].module = vertShaderModule;
    stages[0].pName = "main";
    stages[1].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    stages[1].stage = VK_SHADER_STAGE_FRAGMENT_BIT;
    stages[1].module = tnrFragModule;
    stages[1].pName = "main";

    VkPipelineVertexInputStateCreateInfo vertexInputInfo{};
    vertexInputInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
    VkPipelineInputAssemblyStateCreateInfo inputAssembly{};
    inputAssembly.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
    inputAssembly.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
    VkPipelineViewportStateCreateInfo viewportState{};
    viewportState.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
    viewportState.viewportCount = 1;
    viewportState.scissorCount = 1;
    VkPipelineRasterizationStateCreateInfo rasterizer{};
    rasterizer.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
    rasterizer.polygonMode = VK_POLYGON_MODE_FILL;
    rasterizer.lineWidth = 1.0f;
    rasterizer.cullMode = VK_CULL_MODE_BACK_BIT;
    rasterizer.frontFace = VK_FRONT_FACE_CLOCKWISE;
    VkPipelineMultisampleStateCreateInfo multisampling{};
    multisampling.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
    multisampling.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

    VkPipelineColorBlendAttachmentState blendAttachments[3]{};
    blendAttachments[0].colorWriteMask = 0xF;
    blendAttachments[1].colorWriteMask = 0xF;
    blendAttachments[2].colorWriteMask = 0xF;
    VkPipelineColorBlendStateCreateInfo colorBlending{};
    colorBlending.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
    colorBlending.attachmentCount = 3;
    colorBlending.pAttachments = blendAttachments;

    VkDynamicState dynamicStates[] = {VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR};
    VkPipelineDynamicStateCreateInfo dynamicState{};
    dynamicState.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
    dynamicState.dynamicStateCount = 2;
    dynamicState.pDynamicStates = dynamicStates;

    VkPipelineLayoutCreateInfo pipelineLayoutInfo{};
    pipelineLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    pipelineLayoutInfo.setLayoutCount = 1;
    pipelineLayoutInfo.pSetLayouts = &tnrDescriptorSetLayout;

    if (vkCreatePipelineLayout(device, &pipelineLayoutInfo, nullptr, &tnrPipelineLayout) != VK_SUCCESS) {
        throw std::runtime_error("failed to create TNR pipeline layout!");
    }

    VkGraphicsPipelineCreateInfo pipelineInfo{};
    pipelineInfo.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
    pipelineInfo.stageCount = 2;
    pipelineInfo.pStages = stages;
    pipelineInfo.pVertexInputState = &vertexInputInfo;
    pipelineInfo.pInputAssemblyState = &inputAssembly;
    pipelineInfo.pViewportState = &viewportState;
    pipelineInfo.pRasterizationState = &rasterizer;
    pipelineInfo.pMultisampleState = &multisampling;
    pipelineInfo.pColorBlendState = &colorBlending;
    pipelineInfo.pDynamicState = &dynamicState;
    pipelineInfo.layout = tnrPipelineLayout;
    pipelineInfo.renderPass = tnrRenderPass;
    pipelineInfo.subpass = 0;

    if (vkCreateGraphicsPipelines(device, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &tnrPipeline) != VK_SUCCESS) {
        throw std::runtime_error("failed to create TNR graphics pipeline!");
    }

    vkDestroyShaderModule(device, tnrFragModule, nullptr);
    vkDestroyShaderModule(device, vertShaderModule, nullptr);
}

void VulkanRenderer::createSNRResources() {
    // 1. Render Pass
    VkAttachmentDescription colorAttachment{};
    colorAttachment.format = VK_FORMAT_R16G16B16A16_SFLOAT;
    colorAttachment.samples = VK_SAMPLE_COUNT_1_BIT;
    colorAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
    colorAttachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    colorAttachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    colorAttachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    colorAttachment.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    colorAttachment.finalLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

    VkAttachmentReference colorReference = {0, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL};

    VkSubpassDescription subpass{};
    subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
    subpass.colorAttachmentCount = 1;
    subpass.pColorAttachments = &colorReference;

    VkRenderPassCreateInfo renderPassInfo{};
    renderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
    renderPassInfo.attachmentCount = 1;
    renderPassInfo.pAttachments = &colorAttachment;
    renderPassInfo.subpassCount = 1;
    renderPassInfo.pSubpasses = &subpass;

    if (vkCreateRenderPass(device, &renderPassInfo, nullptr, &snrRenderPass) != VK_SUCCESS) {
        throw std::runtime_error("failed to create SNR render pass!");
    }

    // 2. Images (Double buffered for flip)
    for (int i = 0; i < 2; i++) {
        createImage(RM_WIDTH, RM_HEIGHT, VK_FORMAT_R16G16B16A16_SFLOAT, VK_IMAGE_TILING_OPTIMAL, VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, snrImages[i], snrImageMemories[i]);
        snrImageViews[i] = createImageView(snrImages[i], VK_FORMAT_R16G16B16A16_SFLOAT);
        transitionImageLayout(snrImages[i], VK_FORMAT_R16G16B16A16_SFLOAT, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);

        VkFramebufferCreateInfo framebufferInfo{};
        framebufferInfo.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
        framebufferInfo.renderPass = snrRenderPass;
        framebufferInfo.attachmentCount = 1;
        framebufferInfo.pAttachments = &snrImageViews[i];
        framebufferInfo.width = RM_WIDTH;
        framebufferInfo.height = RM_HEIGHT;
        framebufferInfo.layers = 1;

        if (vkCreateFramebuffer(device, &framebufferInfo, nullptr, &snrFramebuffers[i]) != VK_SUCCESS) {
            throw std::runtime_error("failed to create SNR framebuffer!");
        }
    }

    // 3. Descriptor Set Layout
    VkDescriptorSetLayoutBinding bindings[3]{};
    for(int i=0; i<3; i++) {
        bindings[i].binding = i;
        bindings[i].descriptorCount = 1;
        bindings[i].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        bindings[i].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
    }

    VkDescriptorSetLayoutCreateInfo layoutInfo{};
    layoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    layoutInfo.bindingCount = 3;
    layoutInfo.pBindings = bindings;

    if (vkCreateDescriptorSetLayout(device, &layoutInfo, nullptr, &snrDescriptorSetLayout) != VK_SUCCESS) {
        throw std::runtime_error("failed to create SNR descriptor set layout!");
    }

    // 4. Pipeline
    auto snrFragCode = readFile(std::string(SHADER_DIR) + "/SNR.frag.spv");
    VkShaderModule snrFragModule = createShaderModule(snrFragCode);
    
    auto vertShaderCode = readFile(std::string(SHADER_DIR) + "/shader.vert.spv");
    VkShaderModule vertShaderModule = createShaderModule(vertShaderCode);

    VkPipelineShaderStageCreateInfo stages[2]{};
    stages[0].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    stages[0].stage = VK_SHADER_STAGE_VERTEX_BIT;
    stages[0].module = vertShaderModule;
    stages[0].pName = "main";
    stages[1].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    stages[1].stage = VK_SHADER_STAGE_FRAGMENT_BIT;
    stages[1].module = snrFragModule;
    stages[1].pName = "main";

    VkPipelineVertexInputStateCreateInfo vertexInputInfo{};
    vertexInputInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
    VkPipelineInputAssemblyStateCreateInfo inputAssembly{};
    inputAssembly.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
    inputAssembly.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
    VkPipelineViewportStateCreateInfo viewportState{};
    viewportState.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
    viewportState.viewportCount = 1;
    viewportState.scissorCount = 1;
    VkPipelineRasterizationStateCreateInfo rasterizer{};
    rasterizer.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
    rasterizer.polygonMode = VK_POLYGON_MODE_FILL;
    rasterizer.lineWidth = 1.0f;
    rasterizer.cullMode = VK_CULL_MODE_BACK_BIT;
    rasterizer.frontFace = VK_FRONT_FACE_CLOCKWISE;
    VkPipelineMultisampleStateCreateInfo multisampling{};
    multisampling.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
    multisampling.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

    VkPipelineColorBlendAttachmentState blendAttachment{};
    blendAttachment.colorWriteMask = 0xF;
    VkPipelineColorBlendStateCreateInfo colorBlending{};
    colorBlending.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
    colorBlending.attachmentCount = 1;
    colorBlending.pAttachments = &blendAttachment;

    VkDynamicState dynamicStates[] = {VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR};
    VkPipelineDynamicStateCreateInfo dynamicState{};
    dynamicState.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
    dynamicState.dynamicStateCount = 2;
    dynamicState.pDynamicStates = dynamicStates;

    VkPipelineLayoutCreateInfo pipelineLayoutInfo{};
    pipelineLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    pipelineLayoutInfo.setLayoutCount = 1;
    pipelineLayoutInfo.pSetLayouts = &snrDescriptorSetLayout;

    if (vkCreatePipelineLayout(device, &pipelineLayoutInfo, nullptr, &snrPipelineLayout) != VK_SUCCESS) {
        throw std::runtime_error("failed to create SNR pipeline layout!");
    }

    VkGraphicsPipelineCreateInfo pipelineInfo{};
    pipelineInfo.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
    pipelineInfo.stageCount = 2;
    pipelineInfo.pStages = stages;
    pipelineInfo.pVertexInputState = &vertexInputInfo;
    pipelineInfo.pInputAssemblyState = &inputAssembly;
    pipelineInfo.pViewportState = &viewportState;
    pipelineInfo.pRasterizationState = &rasterizer;
    pipelineInfo.pMultisampleState = &multisampling;
    pipelineInfo.pColorBlendState = &colorBlending;
    pipelineInfo.pDynamicState = &dynamicState;
    pipelineInfo.layout = snrPipelineLayout;
    pipelineInfo.renderPass = snrRenderPass;
    pipelineInfo.subpass = 0;

    if (vkCreateGraphicsPipelines(device, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &snrPipeline) != VK_SUCCESS) {
        throw std::runtime_error("failed to create SNR graphics pipeline!");
    }

    vkDestroyShaderModule(device, snrFragModule, nullptr);
    vkDestroyShaderModule(device, vertShaderModule, nullptr);
}

void VulkanRenderer::createTNRDescriptorSets() {
    uint32_t setCount = MAX_FRAMES_IN_FLIGHT * 2;
    std::vector<VkDescriptorSetLayout> layouts(setCount, tnrDescriptorSetLayout);
    VkDescriptorSetAllocateInfo allocInfo{};
    allocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    allocInfo.descriptorPool = descriptorPool;
    allocInfo.descriptorSetCount = setCount;
    allocInfo.pSetLayouts = layouts.data();

    tnrDescriptorSets.resize(setCount);
    if (vkAllocateDescriptorSets(device, &allocInfo, tnrDescriptorSets.data()) != VK_SUCCESS) {
        throw std::runtime_error("failed to allocate TNR descriptor sets!");
    }

    for (uint32_t i = 0; i < setCount; i++) {
        uint32_t historyIdx = i % 2; // Which history to READ from
        
        VkDescriptorImageInfo rmInfo{offscreenSampler, offscreenImageView, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
        VkDescriptorImageInfo dsInfo{depthTextureSampler, depthDSImageView, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
        VkDescriptorImageInfo mvInfo{mvTextureSampler, mvTextureImageView, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
        // History color comes from SNR output
        VkDescriptorImageInfo prevColorInfo{offscreenSampler, snrImageViews[historyIdx], VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
        VkDescriptorImageInfo prevInfoInfo{offscreenSampler, tnrInfoImageViews[historyIdx], VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
        VkDescriptorImageInfo colorInfo{textureSampler, textureImageView, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};

        VkWriteDescriptorSet writes[6]{};
        for(int j=0; j<6; j++) {
            writes[j].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            writes[j].dstSet = tnrDescriptorSets[i];
            writes[j].dstBinding = j;
            writes[j].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            writes[j].descriptorCount = 1;
        }
        writes[0].pImageInfo = &rmInfo;
        writes[1].pImageInfo = &dsInfo;
        writes[2].pImageInfo = &mvInfo;
        writes[3].pImageInfo = &prevColorInfo;
        writes[4].pImageInfo = &prevInfoInfo;
        writes[5].pImageInfo = &colorInfo;

        vkUpdateDescriptorSets(device, 6, writes, 0, nullptr);
    }
}

void VulkanRenderer::createSNRDescriptorSets() {
    uint32_t setCount = MAX_FRAMES_IN_FLIGHT;
    std::vector<VkDescriptorSetLayout> layouts(setCount, snrDescriptorSetLayout);
    VkDescriptorSetAllocateInfo allocInfo{};
    allocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    allocInfo.descriptorPool = descriptorPool;
    allocInfo.descriptorSetCount = setCount;
    allocInfo.pSetLayouts = layouts.data();

    snrDescriptorSets.resize(setCount);
    if (vkAllocateDescriptorSets(device, &allocInfo, snrDescriptorSets.data()) != VK_SUCCESS) {
        throw std::runtime_error("failed to allocate SNR descriptor sets!");
    }

    for (uint32_t i = 0; i < setCount; i++) {
        VkDescriptorImageInfo tnrOutInfo{offscreenSampler, tnrIntermediateColorImageView, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
        VkDescriptorImageInfo metaInfo{depthTextureSampler, depthDSImageView, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
        // TNR writes to '1 - historyIdx'. So SNR reads from '1 - historyIdx' of current frame.
        // Assuming 'i' is current frame index.
        uint32_t currentHistoryIdx = i % 2; 
        uint32_t readIdx = 1 - currentHistoryIdx;
        VkDescriptorImageInfo tnrAuxInfo{offscreenSampler, tnrInfoImageViews[readIdx], VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};

        VkWriteDescriptorSet writes[3]{};
        
        writes[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[0].dstSet = snrDescriptorSets[i];
        writes[0].dstBinding = 0;
        writes[0].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        writes[0].descriptorCount = 1;
        writes[0].pImageInfo = &tnrOutInfo;

        writes[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[1].dstSet = snrDescriptorSets[i];
        writes[1].dstBinding = 1;
        writes[1].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        writes[1].descriptorCount = 1;
        writes[1].pImageInfo = &metaInfo;

        writes[2].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[2].dstSet = snrDescriptorSets[i];
        writes[2].dstBinding = 2;
        writes[2].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        writes[2].descriptorCount = 1;
        writes[2].pImageInfo = &tnrAuxInfo;

        vkUpdateDescriptorSets(device, 3, writes, 0, nullptr);
    }
}

void VulkanRenderer::createSNR2Resources() {
    // 1. Render Pass
    VkAttachmentDescription colorAttachment{};
    colorAttachment.format = VK_FORMAT_R16G16B16A16_SFLOAT;
    colorAttachment.samples = VK_SAMPLE_COUNT_1_BIT;
    colorAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
    colorAttachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    colorAttachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    colorAttachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    colorAttachment.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    colorAttachment.finalLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

    VkAttachmentReference colorReference = {0, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL};

    VkSubpassDescription subpass{};
    subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
    subpass.colorAttachmentCount = 1;
    subpass.pColorAttachments = &colorReference;

    VkRenderPassCreateInfo renderPassInfo{};
    renderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
    renderPassInfo.attachmentCount = 1;
    renderPassInfo.pAttachments = &colorAttachment;
    renderPassInfo.subpassCount = 1;
    renderPassInfo.pSubpasses = &subpass;

    if (vkCreateRenderPass(device, &renderPassInfo, nullptr, &snr2RenderPass) != VK_SUCCESS) {
        throw std::runtime_error("failed to create SNR2 render pass!");
    }

    // 2. Images
    for (int i = 0; i < 2; i++) {
        createImage(RM_WIDTH, RM_HEIGHT, VK_FORMAT_R16G16B16A16_SFLOAT, VK_IMAGE_TILING_OPTIMAL, VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, snr2Images[i], snr2ImageMemories[i]);
        snr2ImageViews[i] = createImageView(snr2Images[i], VK_FORMAT_R16G16B16A16_SFLOAT);
        transitionImageLayout(snr2Images[i], VK_FORMAT_R16G16B16A16_SFLOAT, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);

        VkFramebufferCreateInfo framebufferInfo{};
        framebufferInfo.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
        framebufferInfo.renderPass = snr2RenderPass;
        framebufferInfo.attachmentCount = 1;
        framebufferInfo.pAttachments = &snr2ImageViews[i];
        framebufferInfo.width = RM_WIDTH;
        framebufferInfo.height = RM_HEIGHT;
        framebufferInfo.layers = 1;

        if (vkCreateFramebuffer(device, &framebufferInfo, nullptr, &snr2Framebuffers[i]) != VK_SUCCESS) {
            throw std::runtime_error("failed to create SNR2 framebuffer!");
        }
    }

    // 3. Descriptor Set Layout
    VkDescriptorSetLayoutBinding binding{};
    binding.binding = 0;
    binding.descriptorCount = 1;
    binding.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    binding.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;

    VkDescriptorSetLayoutCreateInfo layoutInfo{};
    layoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    layoutInfo.bindingCount = 1;
    layoutInfo.pBindings = &binding;

    if (vkCreateDescriptorSetLayout(device, &layoutInfo, nullptr, &snr2DescriptorSetLayout) != VK_SUCCESS) {
        throw std::runtime_error("failed to create SNR2 descriptor set layout!");
    }

    // 4. Pipeline
    auto snr2FragCode = readFile(std::string(SHADER_DIR) + "/SNR2.frag.spv");
    VkShaderModule snr2FragModule = createShaderModule(snr2FragCode);
    
    auto vertShaderCode = readFile(std::string(SHADER_DIR) + "/shader.vert.spv");
    VkShaderModule vertShaderModule = createShaderModule(vertShaderCode);

    VkPipelineShaderStageCreateInfo stages[2]{};
    stages[0].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    stages[0].stage = VK_SHADER_STAGE_VERTEX_BIT;
    stages[0].module = vertShaderModule;
    stages[0].pName = "main";
    stages[1].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    stages[1].stage = VK_SHADER_STAGE_FRAGMENT_BIT;
    stages[1].module = snr2FragModule;
    stages[1].pName = "main";

    VkPipelineVertexInputStateCreateInfo vertexInputInfo{};
    vertexInputInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
    VkPipelineInputAssemblyStateCreateInfo inputAssembly{};
    inputAssembly.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
    inputAssembly.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
    VkPipelineViewportStateCreateInfo viewportState{};
    viewportState.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
    viewportState.viewportCount = 1;
    viewportState.scissorCount = 1;
    VkPipelineRasterizationStateCreateInfo rasterizer{};
    rasterizer.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
    rasterizer.polygonMode = VK_POLYGON_MODE_FILL;
    rasterizer.lineWidth = 1.0f;
    rasterizer.cullMode = VK_CULL_MODE_BACK_BIT;
    rasterizer.frontFace = VK_FRONT_FACE_CLOCKWISE;
    VkPipelineMultisampleStateCreateInfo multisampling{};
    multisampling.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
    multisampling.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;
    VkPipelineColorBlendAttachmentState blendAttachment{};
    blendAttachment.colorWriteMask = 0xF;
    VkPipelineColorBlendStateCreateInfo colorBlending{};
    colorBlending.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
    colorBlending.attachmentCount = 1;
    colorBlending.pAttachments = &blendAttachment;
    VkDynamicState dynamicStates[] = {VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR};
    VkPipelineDynamicStateCreateInfo dynamicState{};
    dynamicState.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
    dynamicState.dynamicStateCount = 2;
    dynamicState.pDynamicStates = dynamicStates;

    VkPipelineLayoutCreateInfo pipelineLayoutInfo{};
    pipelineLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    pipelineLayoutInfo.setLayoutCount = 1;
    pipelineLayoutInfo.pSetLayouts = &snr2DescriptorSetLayout;

    if (vkCreatePipelineLayout(device, &pipelineLayoutInfo, nullptr, &snr2PipelineLayout) != VK_SUCCESS) {
        throw std::runtime_error("failed to create SNR2 pipeline layout!");
    }

    VkGraphicsPipelineCreateInfo pipelineInfo{};
    pipelineInfo.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
    pipelineInfo.stageCount = 2;
    pipelineInfo.pStages = stages;
    pipelineInfo.pVertexInputState = &vertexInputInfo;
    pipelineInfo.pInputAssemblyState = &inputAssembly;
    pipelineInfo.pViewportState = &viewportState;
    pipelineInfo.pRasterizationState = &rasterizer;
    pipelineInfo.pMultisampleState = &multisampling;
    pipelineInfo.pColorBlendState = &colorBlending;
    pipelineInfo.pDynamicState = &dynamicState;
    pipelineInfo.layout = snr2PipelineLayout;
    pipelineInfo.renderPass = snr2RenderPass;
    pipelineInfo.subpass = 0;

    if (vkCreateGraphicsPipelines(device, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &snr2Pipeline) != VK_SUCCESS) {
        throw std::runtime_error("failed to create SNR2 graphics pipeline!");
    }

    vkDestroyShaderModule(device, snr2FragModule, nullptr);
    vkDestroyShaderModule(device, vertShaderModule, nullptr);
}

void VulkanRenderer::createSNR2DescriptorSets() {
    uint32_t setCount = MAX_FRAMES_IN_FLIGHT;
    std::vector<VkDescriptorSetLayout> layouts(setCount, snr2DescriptorSetLayout);
    VkDescriptorSetAllocateInfo allocInfo{};
    allocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    allocInfo.descriptorPool = descriptorPool;
    allocInfo.descriptorSetCount = setCount;
    allocInfo.pSetLayouts = layouts.data();

    snr2DescriptorSets.resize(setCount);
    if (vkAllocateDescriptorSets(device, &allocInfo, snr2DescriptorSets.data()) != VK_SUCCESS) {
        throw std::runtime_error("failed to allocate SNR2 descriptor sets!");
    }

    for (uint32_t i = 0; i < setCount; i++) {
        // Initial binding, will be updated in drawFrame
        VkDescriptorImageInfo snrInfo{offscreenSampler, snrImageViews[0], VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
        VkWriteDescriptorSet write{};
        write.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        write.dstSet = snr2DescriptorSets[i];
        write.dstBinding = 0;
        write.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        write.descriptorCount = 1;
        write.pImageInfo = &snrInfo;

        vkUpdateDescriptorSets(device, 1, &write, 0, nullptr);
    }
}

void VulkanRenderer::createComputeFresnelResources() {
    // 1. Render Pass
    VkAttachmentDescription colorAttachment{};
    colorAttachment.format = VK_FORMAT_R16G16B16A16_SFLOAT;
    colorAttachment.samples = VK_SAMPLE_COUNT_1_BIT;
    colorAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
    colorAttachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    colorAttachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    colorAttachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    colorAttachment.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    colorAttachment.finalLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

    VkAttachmentReference colorReference = {0, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL};

    VkSubpassDescription subpass{};
    subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
    subpass.colorAttachmentCount = 1;
    subpass.pColorAttachments = &colorReference;

    VkRenderPassCreateInfo renderPassInfo{};
    renderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
    renderPassInfo.attachmentCount = 1;
    renderPassInfo.pAttachments = &colorAttachment;
    renderPassInfo.subpassCount = 1;
    renderPassInfo.pSubpasses = &subpass;

    if (vkCreateRenderPass(device, &renderPassInfo, nullptr, &computeFresnelRenderPass) != VK_SUCCESS) {
        throw std::runtime_error("failed to create ComputeFresnel render pass!");
    }

    // 2. Images
    createImage(RM_WIDTH, RM_HEIGHT, VK_FORMAT_R16G16B16A16_SFLOAT, VK_IMAGE_TILING_OPTIMAL, VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, fresnelImage, fresnelImageMemory);
    fresnelImageView = createImageView(fresnelImage, VK_FORMAT_R16G16B16A16_SFLOAT);
    transitionImageLayout(fresnelImage, VK_FORMAT_R16G16B16A16_SFLOAT, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);

    VkFramebufferCreateInfo framebufferInfo{};
    framebufferInfo.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
    framebufferInfo.renderPass = computeFresnelRenderPass;
    framebufferInfo.attachmentCount = 1;
    framebufferInfo.pAttachments = &fresnelImageView;
    framebufferInfo.width = RM_WIDTH;
    framebufferInfo.height = RM_HEIGHT;
    framebufferInfo.layers = 1;

    if (vkCreateFramebuffer(device, &framebufferInfo, nullptr, &computeFresnelFramebuffer) != VK_SUCCESS) {
        throw std::runtime_error("failed to create ComputeFresnel framebuffer!");
    }

    // 3. Descriptor Set Layout
    VkDescriptorSetLayoutBinding bindings[2]{};
    bindings[0].binding = 0;
    bindings[0].descriptorCount = 1;
    bindings[0].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    bindings[0].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;

    bindings[1].binding = 1;
    bindings[1].descriptorCount = 1;
    bindings[1].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    bindings[1].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;

    VkDescriptorSetLayoutCreateInfo layoutInfo{};
    layoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    layoutInfo.bindingCount = 2;
    layoutInfo.pBindings = bindings;

    if (vkCreateDescriptorSetLayout(device, &layoutInfo, nullptr, &computeFresnelDescriptorSetLayout) != VK_SUCCESS) {
        throw std::runtime_error("failed to create ComputeFresnel descriptor set layout!");
    }

    // 4. Pipeline
    auto fragCode = readFile(std::string(SHADER_DIR) + "/computeFresnel.frag.spv");
    VkShaderModule fragModule = createShaderModule(fragCode);
    
    auto vertShaderCode = readFile(std::string(SHADER_DIR) + "/shader.vert.spv");
    VkShaderModule vertShaderModule = createShaderModule(vertShaderCode);

    VkPipelineShaderStageCreateInfo stages[2]{};
    stages[0].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    stages[0].stage = VK_SHADER_STAGE_VERTEX_BIT;
    stages[0].module = vertShaderModule;
    stages[0].pName = "main";
    stages[1].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    stages[1].stage = VK_SHADER_STAGE_FRAGMENT_BIT;
    stages[1].module = fragModule;
    stages[1].pName = "main";

    VkPipelineVertexInputStateCreateInfo vertexInputInfo{};
    vertexInputInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
    VkPipelineInputAssemblyStateCreateInfo inputAssembly{};
    inputAssembly.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
    inputAssembly.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
    VkPipelineViewportStateCreateInfo viewportState{};
    viewportState.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
    viewportState.viewportCount = 1;
    viewportState.scissorCount = 1;
    VkPipelineRasterizationStateCreateInfo rasterizer{};
    rasterizer.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
    rasterizer.polygonMode = VK_POLYGON_MODE_FILL;
    rasterizer.lineWidth = 1.0f;
    rasterizer.cullMode = VK_CULL_MODE_BACK_BIT;
    rasterizer.frontFace = VK_FRONT_FACE_CLOCKWISE;
    VkPipelineMultisampleStateCreateInfo multisampling{};
    multisampling.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
    multisampling.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

    VkPipelineColorBlendAttachmentState blendAttachment{};
    blendAttachment.colorWriteMask = 0xF;
    VkPipelineColorBlendStateCreateInfo colorBlending{};
    colorBlending.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
    colorBlending.attachmentCount = 1;
    colorBlending.pAttachments = &blendAttachment;

    VkDynamicState dynamicStates[] = {VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR};
    VkPipelineDynamicStateCreateInfo dynamicState{};
    dynamicState.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
    dynamicState.dynamicStateCount = 2;
    dynamicState.pDynamicStates = dynamicStates;

    VkPipelineLayoutCreateInfo pipelineLayoutInfo{};
    pipelineLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    pipelineLayoutInfo.setLayoutCount = 1;
    pipelineLayoutInfo.pSetLayouts = &computeFresnelDescriptorSetLayout;

    if (vkCreatePipelineLayout(device, &pipelineLayoutInfo, nullptr, &computeFresnelPipelineLayout) != VK_SUCCESS) {
        throw std::runtime_error("failed to create ComputeFresnel pipeline layout!");
    }

    VkGraphicsPipelineCreateInfo pipelineInfo{};
    pipelineInfo.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
    pipelineInfo.stageCount = 2;
    pipelineInfo.pStages = stages;
    pipelineInfo.pVertexInputState = &vertexInputInfo;
    pipelineInfo.pInputAssemblyState = &inputAssembly;
    pipelineInfo.pViewportState = &viewportState;
    pipelineInfo.pRasterizationState = &rasterizer;
    pipelineInfo.pMultisampleState = &multisampling;
    pipelineInfo.pColorBlendState = &colorBlending;
    pipelineInfo.pDynamicState = &dynamicState;
    pipelineInfo.layout = computeFresnelPipelineLayout;
    pipelineInfo.renderPass = computeFresnelRenderPass;
    pipelineInfo.subpass = 0;

    if (vkCreateGraphicsPipelines(device, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &computeFresnelPipeline) != VK_SUCCESS) {
        throw std::runtime_error("failed to create ComputeFresnel graphics pipeline!");
    }

    vkDestroyShaderModule(device, fragModule, nullptr);
    vkDestroyShaderModule(device, vertShaderModule, nullptr);
}

void VulkanRenderer::createComputeFresnelDescriptorSets() {
    std::vector<VkDescriptorSetLayout> layouts(MAX_FRAMES_IN_FLIGHT, computeFresnelDescriptorSetLayout);
    VkDescriptorSetAllocateInfo allocInfo{};
    allocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    allocInfo.descriptorPool = descriptorPool;
    allocInfo.descriptorSetCount = MAX_FRAMES_IN_FLIGHT;
    allocInfo.pSetLayouts = layouts.data();

    computeFresnelDescriptorSets.resize(MAX_FRAMES_IN_FLIGHT);
    if (vkAllocateDescriptorSets(device, &allocInfo, computeFresnelDescriptorSets.data()) != VK_SUCCESS) {
        throw std::runtime_error("failed to allocate ComputeFresnel descriptor sets!");
    }

    for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
        VkDescriptorImageInfo depthInfo{depthTextureSampler, depthTextureImageView, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
        VkDescriptorImageInfo normalInfo{normalTextureSampler, normalTextureImageView, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};

        VkWriteDescriptorSet descriptorWrites[2]{};

        descriptorWrites[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        descriptorWrites[0].dstSet = computeFresnelDescriptorSets[i];
        descriptorWrites[0].dstBinding = 0;
        descriptorWrites[0].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        descriptorWrites[0].descriptorCount = 1;
        descriptorWrites[0].pImageInfo = &depthInfo;
        
        descriptorWrites[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        descriptorWrites[1].dstSet = computeFresnelDescriptorSets[i];
        descriptorWrites[1].dstBinding = 1;
        descriptorWrites[1].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        descriptorWrites[1].descriptorCount = 1;
        descriptorWrites[1].pImageInfo = &normalInfo;

        vkUpdateDescriptorSets(device, 2, descriptorWrites, 0, nullptr);
    }
}

void VulkanRenderer::createTNR2Resources() {
    // 1. Render Pass
    VkAttachmentDescription colorAttachment{};
    colorAttachment.format = VK_FORMAT_R16G16B16A16_SFLOAT;
    colorAttachment.samples = VK_SAMPLE_COUNT_1_BIT;
    colorAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
    colorAttachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    colorAttachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    colorAttachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    colorAttachment.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    colorAttachment.finalLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

    VkAttachmentDescription infoAttachment{};
    infoAttachment.format = VK_FORMAT_R16G16B16A16_SFLOAT;
    infoAttachment.samples = VK_SAMPLE_COUNT_1_BIT;
    infoAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
    infoAttachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    infoAttachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    infoAttachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    infoAttachment.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    infoAttachment.finalLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

    VkAttachmentReference colorReference = {0, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL};
    VkAttachmentReference infoReference = {1, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL};
    VkAttachmentReference attachmentsRef[] = {colorReference, infoReference};

    VkSubpassDescription subpass{};
    subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
    subpass.colorAttachmentCount = 2;
    subpass.pColorAttachments = attachmentsRef;

    VkAttachmentDescription attachments[] = {colorAttachment, infoAttachment};
    VkRenderPassCreateInfo renderPassInfo{};
    renderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
    renderPassInfo.attachmentCount = 2;
    renderPassInfo.pAttachments = attachments;
    renderPassInfo.subpassCount = 1;
    renderPassInfo.pSubpasses = &subpass;

    if (vkCreateRenderPass(device, &renderPassInfo, nullptr, &tnr2RenderPass) != VK_SUCCESS) {
        throw std::runtime_error("failed to create TNR2 render pass!");
    }

    // 2. Images
    for (int i = 0; i < 2; i++) {
        // Color
        createImage(RM_WIDTH, RM_HEIGHT, VK_FORMAT_R16G16B16A16_SFLOAT, VK_IMAGE_TILING_OPTIMAL, VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, tnr2Images[i], tnr2ImageMemories[i]);
        tnr2ImageViews[i] = createImageView(tnr2Images[i], VK_FORMAT_R16G16B16A16_SFLOAT);
        transitionImageLayout(tnr2Images[i], VK_FORMAT_R16G16B16A16_SFLOAT, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);

        // Info
        createImage(RM_WIDTH, RM_HEIGHT, VK_FORMAT_R16G16B16A16_SFLOAT, VK_IMAGE_TILING_OPTIMAL, VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, tnr2InfoImages[i], tnr2InfoImageMemories[i]);
        tnr2InfoImageViews[i] = createImageView(tnr2InfoImages[i], VK_FORMAT_R16G16B16A16_SFLOAT);
        transitionImageLayout(tnr2InfoImages[i], VK_FORMAT_R16G16B16A16_SFLOAT, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);

        VkImageView attachmentsFB[] = {tnr2ImageViews[i], tnr2InfoImageViews[i]};
        VkFramebufferCreateInfo framebufferInfo{};
        framebufferInfo.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
        framebufferInfo.renderPass = tnr2RenderPass;
        framebufferInfo.attachmentCount = 2;
        framebufferInfo.pAttachments = attachmentsFB;
        framebufferInfo.width = RM_WIDTH;
        framebufferInfo.height = RM_HEIGHT;
        framebufferInfo.layers = 1;

        if (vkCreateFramebuffer(device, &framebufferInfo, nullptr, &tnr2Framebuffers[i]) != VK_SUCCESS) {
            throw std::runtime_error("failed to create TNR2 framebuffer!");
        }
    }

    // 3. Descriptor Set Layout
    VkDescriptorSetLayoutBinding bindings[6]{};
    for(int i=0; i<6; i++) {
        bindings[i].binding = i;
        bindings[i].descriptorCount = 1;
        bindings[i].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        bindings[i].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
    }

    VkDescriptorSetLayoutCreateInfo layoutInfo{};
    layoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    layoutInfo.bindingCount = 6;
    layoutInfo.pBindings = bindings;

    if (vkCreateDescriptorSetLayout(device, &layoutInfo, nullptr, &tnr2DescriptorSetLayout) != VK_SUCCESS) {
        throw std::runtime_error("failed to create TNR2 descriptor set layout!");
    }

    // 4. Pipeline
    auto fragCode = readFile(std::string(SHADER_DIR) + "/TNR2.frag.spv");
    VkShaderModule fragModule = createShaderModule(fragCode);
    
    auto vertShaderCode = readFile(std::string(SHADER_DIR) + "/shader.vert.spv");
    VkShaderModule vertShaderModule = createShaderModule(vertShaderCode);

    VkPipelineShaderStageCreateInfo stages[2]{};
    stages[0].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    stages[0].stage = VK_SHADER_STAGE_VERTEX_BIT;
    stages[0].module = vertShaderModule;
    stages[0].pName = "main";
    stages[1].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    stages[1].stage = VK_SHADER_STAGE_FRAGMENT_BIT;
    stages[1].module = fragModule;
    stages[1].pName = "main";

    VkPipelineVertexInputStateCreateInfo vertexInputInfo{};
    vertexInputInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
    VkPipelineInputAssemblyStateCreateInfo inputAssembly{};
    inputAssembly.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
    inputAssembly.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
    VkPipelineViewportStateCreateInfo viewportState{};
    viewportState.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
    viewportState.viewportCount = 1;
    viewportState.scissorCount = 1;
    VkPipelineRasterizationStateCreateInfo rasterizer{};
    rasterizer.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
    rasterizer.polygonMode = VK_POLYGON_MODE_FILL;
    rasterizer.lineWidth = 1.0f;
    rasterizer.cullMode = VK_CULL_MODE_BACK_BIT;
    rasterizer.frontFace = VK_FRONT_FACE_CLOCKWISE;
    VkPipelineMultisampleStateCreateInfo multisampling{};
    multisampling.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
    multisampling.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

    VkPipelineColorBlendAttachmentState blendAttachments[2]{};
    blendAttachments[0].colorWriteMask = 0xF;
    blendAttachments[1].colorWriteMask = 0xF;
    VkPipelineColorBlendStateCreateInfo colorBlending{};
    colorBlending.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
    colorBlending.attachmentCount = 2;
    colorBlending.pAttachments = blendAttachments;

    VkDynamicState dynamicStates[] = {VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR};
    VkPipelineDynamicStateCreateInfo dynamicState{};
    dynamicState.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
    dynamicState.dynamicStateCount = 2;
    dynamicState.pDynamicStates = dynamicStates;

    VkPipelineLayoutCreateInfo pipelineLayoutInfo{};
    pipelineLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    pipelineLayoutInfo.setLayoutCount = 1;
    pipelineLayoutInfo.pSetLayouts = &tnr2DescriptorSetLayout;

    if (vkCreatePipelineLayout(device, &pipelineLayoutInfo, nullptr, &tnr2PipelineLayout) != VK_SUCCESS) {
        throw std::runtime_error("failed to create TNR2 pipeline layout!");
    }

    VkGraphicsPipelineCreateInfo pipelineInfo{};
    pipelineInfo.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
    pipelineInfo.stageCount = 2;
    pipelineInfo.pStages = stages;
    pipelineInfo.pVertexInputState = &vertexInputInfo;
    pipelineInfo.pInputAssemblyState = &inputAssembly;
    pipelineInfo.pViewportState = &viewportState;
    pipelineInfo.pRasterizationState = &rasterizer;
    pipelineInfo.pMultisampleState = &multisampling;
    pipelineInfo.pColorBlendState = &colorBlending;
    pipelineInfo.pDynamicState = &dynamicState;
    pipelineInfo.layout = tnr2PipelineLayout;
    pipelineInfo.renderPass = tnr2RenderPass;
    pipelineInfo.subpass = 0;

    if (vkCreateGraphicsPipelines(device, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &tnr2Pipeline) != VK_SUCCESS) {
        throw std::runtime_error("failed to create TNR2 graphics pipeline!");
    }

    vkDestroyShaderModule(device, fragModule, nullptr);
    vkDestroyShaderModule(device, vertShaderModule, nullptr);
}


void VulkanRenderer::createTNR2DescriptorSets() {
    uint32_t setCount = MAX_FRAMES_IN_FLIGHT * 2;
    std::vector<VkDescriptorSetLayout> layouts(setCount, tnr2DescriptorSetLayout);
    VkDescriptorSetAllocateInfo allocInfo{};
    allocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    allocInfo.descriptorPool = descriptorPool;
    allocInfo.descriptorSetCount = setCount;
    allocInfo.pSetLayouts = layouts.data();

    tnr2DescriptorSets.resize(setCount);
    if (vkAllocateDescriptorSets(device, &allocInfo, tnr2DescriptorSets.data()) != VK_SUCCESS) {
        throw std::runtime_error("failed to allocate TNR2 descriptor sets!");
    }
    
    for (size_t i = 0; i < setCount; i++) {
        uint32_t historyIdx = i % 2; // Matches tnrHistoryIndex

        // TNR2 reads:
        // SNR2 (Current Input): writes to [1-historyIdx] in previous pass, so read from [1-historyIdx].
        // TNR2 History: read from [historyIdx].
        // TNR Info: read from [1-historyIdx] (Assuming it was written in TNR pass using "1-historyIdx").
        
        VkDescriptorImageInfo snrInfo{offscreenSampler, snr2ImageViews[1 - historyIdx], VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
        VkDescriptorImageInfo historyInfo{offscreenSampler, tnr2ImageViews[historyIdx], VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
        VkDescriptorImageInfo depthInfo{depthTextureSampler, depthTextureImageView, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
        VkDescriptorImageInfo mvInfo{mvTextureSampler, mvTextureImageView, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
        VkDescriptorImageInfo fresnelInfo{offscreenSampler, fresnelImageView, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
        VkDescriptorImageInfo tnrInfoInfo{offscreenSampler, tnrInfoImageViews[1 - historyIdx], VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};

        VkWriteDescriptorSet writes[6]{};
        
        writes[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[0].dstSet = tnr2DescriptorSets[i];
        writes[0].dstBinding = 0; // SNR_out0 (actually SNR2 out)
        writes[0].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        writes[0].descriptorCount = 1;
        writes[0].pImageInfo = &snrInfo;

        writes[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[1].dstSet = tnr2DescriptorSets[i];
        writes[1].dstBinding = 1; // TNR2 History
        writes[1].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        writes[1].descriptorCount = 1;
        writes[1].pImageInfo = &historyInfo;
        
        writes[2].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[2].dstSet = tnr2DescriptorSets[i];
        writes[2].dstBinding = 2; // Depth
        writes[2].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        writes[2].descriptorCount = 1;
        writes[2].pImageInfo = &depthInfo;
        
        writes[3].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[3].dstSet = tnr2DescriptorSets[i];
        writes[3].dstBinding = 3; // MV
        writes[3].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        writes[3].descriptorCount = 1;
        writes[3].pImageInfo = &mvInfo;
        
        writes[4].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[4].dstSet = tnr2DescriptorSets[i];
        writes[4].dstBinding = 4; // Fresnel
        writes[4].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        writes[4].descriptorCount = 1;
        writes[4].pImageInfo = &fresnelInfo;
        
        writes[5].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[5].dstSet = tnr2DescriptorSets[i];
        writes[5].dstBinding = 5; // TNR Info
        writes[5].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        writes[5].descriptorCount = 1;
        writes[5].pImageInfo = &tnrInfoInfo;

        vkUpdateDescriptorSets(device, 6, writes, 0, nullptr);
    }
}
