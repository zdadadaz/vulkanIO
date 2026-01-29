#pragma once

#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>

#include <vector>
#include <string>
#include <iostream>
#include <stdexcept>
#include <cstdlib>
#include <vector>
#include <optional>
#include <set>
#include <algorithm>
#include <fstream>

class VulkanRenderer {
public:
    void run();

private:
    // Window settings
    const uint32_t WIDTH = 1920;
    const uint32_t HEIGHT = 864;
    const uint32_t STRIDE = 1;
    const uint32_t RM_WIDTH = WIDTH / STRIDE;
    const uint32_t RM_HEIGHT = HEIGHT / STRIDE;
    
    // Core Vulkan
    GLFWwindow* window;
    VkInstance instance;
    VkDebugUtilsMessengerEXT debugMessenger;
    VkSurfaceKHR surface;
    VkPhysicalDevice physicalDevice = VK_NULL_HANDLE;
    VkDevice device;
    VkQueue graphicsQueue;
    VkQueue presentQueue;
    uint32_t graphicsQueueFamilyIndex;
    
    // Swapchain
    VkSwapchainKHR swapchain;
    std::vector<VkImage> swapchainImages;
    VkFormat swapchainImageFormat;
    VkExtent2D swapchainExtent;
    std::vector<VkImageView> swapchainImageViews;
    std::vector<VkFramebuffer> swapchainFramebuffers;
    
    // Graphics Pipeline
    VkRenderPass renderPass;
    VkDescriptorSetLayout descriptorSetLayout;
    
    // Command Buffers
    VkCommandPool commandPool;
    std::vector<VkCommandBuffer> commandBuffers;
    
    // Synchronization
    std::vector<VkSemaphore> imageAvailableSemaphores;
    std::vector<VkSemaphore> renderFinishedSemaphores;
    std::vector<VkFence> inFlightFences;
    uint32_t currentFrame = 0;
    const int MAX_FRAMES_IN_FLIGHT = 2;
    
    // Texture Resources
    VkImage textureImage;
    VkDeviceMemory textureImageMemory;
    VkImageView textureImageView;
    VkSampler textureSampler;

    // Depth Texture Resources
    VkImage depthTextureImage;
    VkDeviceMemory depthTextureImageMemory;
    VkImageView depthTextureImageView;
    VkSampler depthTextureSampler;
    
    VkBuffer stagingBuffer;
    VkDeviceMemory stagingBufferMemory;

    VkBuffer depthStagingBuffer;
    VkDeviceMemory depthStagingBufferMemory;

    // Normal Texture Resources
    VkImage normalTextureImage;
    VkDeviceMemory normalTextureImageMemory;
    VkImageView normalTextureImageView;
    VkSampler normalTextureSampler;
    VkBuffer normalStagingBuffer;
    VkDeviceMemory normalStagingBufferMemory;

    // Albedo Texture Resources (New)
    VkImage albedoTextureImage;
    VkDeviceMemory albedoTextureImageMemory;
    VkImageView albedoTextureImageView;
    VkSampler albedoTextureSampler;
    VkBuffer albedoStagingBuffer;
    VkDeviceMemory albedoStagingBufferMemory;
    
    // Offscreen (Low-Res RM)
    VkImage offscreenImage;
    VkDeviceMemory offscreenImageMemory;
    VkImageView offscreenImageView;
    VkSampler offscreenSampler;
    VkRenderPass offscreenRenderPass;
    VkFramebuffer offscreenFramebuffer;
    VkPipeline offscreenPipeline;
    VkPipelineLayout offscreenPipelineLayout;
    
    // Final Pass (Upscale)
    VkDescriptorSetLayout finalDescriptorSetLayout;
    VkPipeline finalPipeline;
    VkPipelineLayout finalPipelineLayout;
    std::vector<VkDescriptorSet> finalDescriptorSets;

    // DepthDS Pass
    VkImage depthDSImage;
    VkDeviceMemory depthDSImageMemory;
    VkImageView depthDSImageView;
    VkRenderPass depthDSRenderPass;
    VkFramebuffer depthDSFramebuffer;
    VkPipeline depthDSPipeline;
    VkPipelineLayout depthDSPipelineLayout;
    VkDescriptorSetLayout depthDSDescriptorSetLayout;
    std::vector<VkDescriptorSet> depthDSDescriptorSets;

    VkDescriptorPool descriptorPool;
    std::vector<VkDescriptorSet> descriptorSets;

    // TNR Pass
    VkRenderPass tnrRenderPass;
    VkPipeline tnrPipeline;
    VkPipelineLayout tnrPipelineLayout;
    VkDescriptorSetLayout tnrDescriptorSetLayout;
    std::vector<VkDescriptorSet> tnrDescriptorSets;
    
    // TNR Textures (Double buffered for feedback)
    VkImage tnrColorImages[2];
    VkDeviceMemory tnrColorImageMemories[2];
    VkImageView tnrColorImageViews[2];
    
    VkImage tnrInfoImages[2];
    VkDeviceMemory tnrInfoImageMemories[2];
    VkImageView tnrInfoImageViews[2];
    
    VkFramebuffer tnrFramebuffers[2];
    uint32_t tnrHistoryIndex = 0;

    // SNR Pass
    VkRenderPass snrRenderPass;
    VkPipeline snrPipeline;
    VkPipelineLayout snrPipelineLayout;
    VkDescriptorSetLayout snrDescriptorSetLayout;
    std::vector<VkDescriptorSet> snrDescriptorSets;

    VkImage tnrIntermediateColorImage;
    VkDeviceMemory tnrIntermediateColorImageMemory;
    VkImageView tnrIntermediateColorImageView;

    VkImage tnrOut2Image;
    VkDeviceMemory tnrOut2ImageMemory;
    VkImageView tnrOut2ImageView;

    VkImage snrImages[2];
    VkDeviceMemory snrImageMemories[2];
    VkImageView snrImageViews[2];
    VkFramebuffer snrFramebuffers[2];

    // SNR2 Pass
    VkRenderPass snr2RenderPass;
    VkPipeline snr2Pipeline;
    VkPipelineLayout snr2PipelineLayout;
    VkDescriptorSetLayout snr2DescriptorSetLayout;
    std::vector<VkDescriptorSet> snr2DescriptorSets;

    VkImage snr2Images[2];
    VkDeviceMemory snr2ImageMemories[2];
    VkImageView snr2ImageViews[2];
    VkFramebuffer snr2Framebuffers[2];

    // MV Texture Resources
    VkImage mvTextureImage;
    VkDeviceMemory mvTextureImageMemory;
    VkImageView mvTextureImageView;
    VkSampler mvTextureSampler;
    VkBuffer mvStagingBuffer;
    VkDeviceMemory mvStagingBufferMemory;

    // Image Sequence Logic
    int currentFrameIndex = 0;
    int frameDelayCounter = 0;
    const int frameDelay = 2; // Simple delay to control playback speed if needed
    
    void initWindow();
    void initVulkan();
    void mainLoop();
    void cleanup();
    
    // Vulkan Initialization Helpers
    void createInstance();
    void setupDebugMessenger();
    void createSurface();
    void pickPhysicalDevice();
    void createLogicalDevice();
    void createSwapchain();
    void createImageViews();
    void createRenderPass();
    void createDescriptorSetLayout();
    void createGraphicsPipeline();
    void createFramebuffers();
    void createCommandPool();
    void createTextureImage(); // Initial dummy/first frame
    void createTextureImageView();
    void createTextureSampler();
    void createDepthTextureImage();
    void createDepthTextureImageView();
    void createDepthTextureSampler();
    void createDescriptorPool();
    void createDescriptorSets();
    void createCommandBuffers();
    void createOffscreenResources();
    void createDepthDSResources();
    void createFinalDescriptorSetLayout();
    void createSyncObjects();
    
    void createNormalTextureImage();
    void createNormalTextureImageView();
    void createNormalTextureSampler();

    void createAlbedoTextureImage();
    void createAlbedoTextureImageView();
    void createAlbedoTextureSampler();

    void createMVTextureImage();
    void createMVTextureImageView();
    void createMVTextureSampler();

    void createTNRResources();
    void createTNRDescriptorSets();

    void createSNRResources();
    void createSNRDescriptorSets();

    void createSNR2Resources();
    void createSNR2DescriptorSets();

    // Rendering
    void drawFrame();
    void recordCommandBuffer(VkCommandBuffer commandBuffer, uint32_t imageIndex);
    
    // Texture Updating
    void updateTexture();
    void loadRawImage(const std::string& filename, void* pixels, const std::string& fallbackPrefix = "");
    
    // Helpers
    bool checkValidationLayerSupport();
    std::vector<const char*> getRequiredExtensions();
    uint32_t findMemoryType(uint32_t typeFilter, VkMemoryPropertyFlags properties);
    void createBuffer(VkDeviceSize size, VkBufferUsageFlags usage, VkMemoryPropertyFlags properties, VkBuffer& buffer, VkDeviceMemory& bufferMemory);
    void createImage(uint32_t width, uint32_t height, VkFormat format, VkImageTiling tiling, VkImageUsageFlags usage, VkMemoryPropertyFlags properties, VkImage& image, VkDeviceMemory& imageMemory);
    VkImageView createImageView(VkImage image, VkFormat format);
    void transitionImageLayout(VkImage image, VkFormat format, VkImageLayout oldLayout, VkImageLayout newLayout);
    void copyBufferToImage(VkBuffer buffer, VkImage image, uint32_t width, uint32_t height);
    VkShaderModule createShaderModule(const std::vector<char>& code);
    
    static std::vector<char> readFile(const std::string& filename);
    static VKAPI_ATTR VkBool32 VKAPI_CALL debugCallback(VkDebugUtilsMessageSeverityFlagBitsEXT messageSeverity, VkDebugUtilsMessageTypeFlagsEXT messageType, const VkDebugUtilsMessengerCallbackDataEXT* pCallbackData, void* pUserData);
};
